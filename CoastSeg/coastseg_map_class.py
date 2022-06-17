import os
from ipyleaflet import DrawControl, GeoJSON, LayersControl
import leafmap
from CoastSeg import download_roi
from CoastSeg import bbox
from CoastSeg import make_overlapping_roi
from CoastSeg import zoo_model_module
from CoastSeg import file_functions


class CoastSeg_Map:
    shoreline_file=os.getcwd()+os.sep+"third_party_data"+os.sep+"stanford-xv279yj9196-geojson.json"
    def __init__(self, map_settings: dict):
        # data : geojson data of the rois generated
        self.data=None
        # selected_set : ids of the selected rois
        self.selected_set=set()
        # geojson_layer : layer with all rois
        self.geojson_layer=None
        # selected layer :  layer containing all selected rois
        self.selected_layer = None
        # shapes_list : Empty list to hold all the polygons drawn by the user
        self.shapes_list=[]
        # coastline_for_map : coastline vector geojson for map layer
        self.coastline_for_map=None
        # selected_ROI : Geojson for all the ROIs selected by the user
        self.selected_ROI=None
        self.collection=None
        CoastSeg_Map.check_shoreline_file_exists()
        self.m = leafmap.Map(
                        draw_control=map_settings["draw_control"],
                        measure_control=map_settings["measure_control"],
                        fullscreen_control=map_settings["fullscreen_control"],
                        attribution_control=map_settings["attribution_control"],
                        center=map_settings["center_point"],
                        zoom=map_settings["zoom"],
                        layout=map_settings["Layout"])
        # # Create drawing controls
        self.draw_control=self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.m.add_control(self.draw_control)
        layer_control = LayersControl(position='topright')
        self.m.add_control(layer_control)
    
    def check_shoreline_file_exists():
        if not os.path.exists(CoastSeg_Map.shoreline_file):
            print("\n The geojson shoreline file does not exist.")
            print("Please ensure the shoreline file is the directory 'third_party_data' ")
    

    def create_DrawControl(self,draw_control):
        draw_control.polyline = {}
        draw_control.circlemarker = {}
        draw_control.polygon = {}
        draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "green",
                "color": "green",
                "fillOpacity": 0.1,
                "Opacity": 0.1
            },
            "drawError": {
                "color": "#dd253b",
                "message": "Ops!"
            },
            "allowIntersection": False,
            "transform":True
        }
        return  draw_control 
   
   
    def handle_draw(self,target, action, geo_json):
        self.action=action
        self.geo_json=geo_json
        self.target=target
        if self.draw_control.last_action == 'created'and self.draw_control.last_draw['geometry']['type']=='Polygon' :
            self.shapes_list.append( self.draw_control.last_draw['geometry'])
        if self.draw_control.last_action == 'deleted':
            self.shapes_list.pop()
    
    
    def set_data(self, roi_filename):
        # Read the geojson for all the ROIs generated
        self.data=download_roi.read_geojson_file(roi_filename)
        # Add style to each feature in the geojson
        for feature in self.data["features"]:
            feature["properties"]["style"] = {
                "color": "grey",
                "weight": 1,
                "fillColor": "grey",
                "fillOpacity": 0.2,
            }
    
    
    def generate_ROIS(self, roi_filename, csv_filename, progressbar=None):
        # Make sure your bounding box is within the allowed size
        bbox.validate_bbox_size(self.shapes_list)
        #dictionary containing geojson coastline
        roi_coastline=bbox.get_coastline(CoastSeg_Map.shoreline_file,self.shapes_list)
        #coastline styled for the map
        self.coastline_for_map=self.get_coastline_layer(roi_coastline)
        self.m.add_layer(self.coastline_for_map)
        #Get the rois using the coastline  within bounding box
        geojson_polygons=make_overlapping_roi.get_ROIs(roi_coastline,roi_filename,csv_filename,progressbar)
        # Save the data from the ROI file to data
        self.set_data(roi_filename)
        # overlap_btw_vectors_df=make_overlapping_roi.min_overlap_btw_vectors(roi_filename,csv_filename,overlap_percent=.65)


    def save_roi_to_file(self, selected_roi_file, roi_filename):
        self.selected_ROI=download_roi.save_roi(roi_filename, selected_roi_file, self.selected_set)
  
  
    def get_coastline_layer(self,roi_coastline: dict):
        """Returns a GeoJSON object that can be added as layer to map """
        assert roi_coastline != {}, "ERROR.\n Empty geojson cannot be drawn onto  map"
        return GeoJSON(
            data=roi_coastline,
            name="Coastline",
            style={
                'color': 'yellow',
                'fill_color': 'yellow',
                'opacity': 1,
                'dashArray': '5',
                'fillOpacity': 0.5,
                'weight': 4},
            hover_style={
                'color': 'white',
                'dashArray': '4',
                'fillOpacity': 0.7},
        )  
    
    
    def get_geojson_layer(self):
        if self.geojson_layer is None:
             self.geojson_layer=GeoJSON(data=self.data, name="geojson data", hover_style={"fillColor": "red"})
        return self.geojson_layer
    
    
    def geojson_onclick_handler(self, event=None, id=None, properties=None, **args):
        if properties is None:
            return
        cid = properties["id"]
        self.selected_set.add(cid)
        if self.selected_layer is not None:
            self.m.remove_layer(self.selected_layer)
            
        self.selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue"},
        )
        self.selected_layer.on_click(self.selected_onclick_handler)
        self.m.add_layer(self.selected_layer)
        
        
    def selected_onclick_handler(self,event=None, id=None, properties=None, **args):
        """This is the on click handler for a layer that is selected.
        This method removes the give layer's cid from the selected_set and removes the layer from
        select_layer."""
        if properties is None:
            return
        # Remove the current layers cid from selected set
        cid = properties["id"]
        self.selected_set.remove(cid)
        if self.selected_layer is not None:
            self.m.remove_layer(self.selected_layer)
        # Recreate the selected layers wihout the layer that was removed
        self.selected_layer = GeoJSON(
            data = self.convert_selected_set_to_geojson(self.selected_set),
            name="Selected ROIs",
            hover_style={"fillColor": "blue"},
        )
        # Recreate the onclick handler for the selected layers
        self.selected_layer.on_click(self.selected_onclick_handler)
        # Add selected layer to the map
        self.m.add_layer(self.selected_layer)
    
    
    def add_geojson_layer_to_map(self):
        geojson_layer =self.get_geojson_layer()
        geojson_layer.on_click(self.geojson_onclick_handler)
        self.m.add_layer(geojson_layer)
        
            
    def convert_selected_set_to_geojson(self,selected_set):
        geojson = {"type": "FeatureCollection", "features": []}
        # Select the geojson in the selected layer
        geojson["features"] = [
            feature
            for feature in self.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        # Modify geojson style for each polygon in the selected layer
        for feature in self.data["features"]:
            feature["properties"]["style"] = {
                "color": "blue",
                "weight": 2,
                "fillColor": "grey",
                "fillOpacity": 0.2,
            }
        return geojson
