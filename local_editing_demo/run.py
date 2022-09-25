"""
Author: Peizhi Yan
Date: Nov-08-2021
-----------------------------------------------
Updates:
* Nov-09-2021
    1) Use ffhq mean shape as default loaded shape 
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
import math
from util.main import *
import os

LABEL_H = 25 # height of label
LABEL_BOLD_H = 26 # height of bold font label
BUTTON_H = 25 # height of button

## Lower and upper bounds of control variables
LB = -3.0
UB = 3.0
STEP = 0.5 # editing step

def rotation_matrix(yaw):
    """ Compute the rotation matrix """
    ## 0 <= yaw <= 3.14 
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]],)
    return Ry


class Application(tk.Frame):
    def __init__(self, root=None):
        super().__init__(root)
        self.root = root
        self.pack()
        self.V = None # vertices
        self.T = None # textures
        self.mesh = None # open3d mesh
        self.o3d_viewer_is_open = False # a flag

        ## Predicted Latent encodings
        self.part_latents = {}

        ## Predicted Offsets
        self.pred_offsets = {}

        ## Rotation Variable
        self.rotate_yaw_angle = tk.DoubleVar()

        ## Controller Variables: Overall
        self.maximum_facial_width = tk.DoubleVar()
        self.madibular_width = tk.DoubleVar()
        self.upper_facial_depth = tk.DoubleVar()
        self.middle_facial_depth = tk.DoubleVar()
        self.lower_facial_depth = tk.DoubleVar()
        self.facial_height = tk.DoubleVar()
        self.upper_facial_height = tk.DoubleVar()
        self.lower_facial_height = tk.DoubleVar()

        ## Controller Variables: Nose
        self.nose_tip_y = tk.DoubleVar() 
        self.nose_tip_z = tk.DoubleVar() 
        self.nose_height = tk.DoubleVar() 
        self.nose_width = tk.DoubleVar() 
        self.tip_width = tk.DoubleVar() 
        self.bridge_width = tk.DoubleVar()

        ## Controller Variables: Eyebrows
        self.front_thickness = tk.DoubleVar() 
        self.tail_thickness = tk.DoubleVar() 
        self.eyebrow_length = tk.DoubleVar() 
        self.curve_strength = tk.DoubleVar() 

        ## Controller Variables: Eyes
        self.pupils_distance = tk.DoubleVar()
        self.eye_height = tk.DoubleVar()
        self.canthus_distance = tk.DoubleVar()
        self.medial_canthus_y = tk.DoubleVar()
        self.lateral_canthus_y = tk.DoubleVar()

        ## Controller Variables: Upper Lip
        self.labial_fissure_width = tk.DoubleVar()
        self.upper_lip_height = tk.DoubleVar()
        self.upper_lip_width = tk.DoubleVar()
        self.upper_lip_end_height = tk.DoubleVar()

        ## Controller Variables: Lower Lip
        self.lower_lip_height = tk.DoubleVar()
        self.lower_lip_width = tk.DoubleVar()
        self.lower_lip_end_height = tk.DoubleVar()


        self.create_GUI()
        #self.load_and_reconstruct()
        self.default_reconstruct()

    def add_editing_scaler(self, scale_name, parent, variable, command):
        # label
        dummy_label = tk.Label(parent, text=scale_name)
        #dummy_label.configure(background='white')
        dummy_label.pack(side=TOP,  fill="x")
        # scaler
        dummy_scale = tk.Scale(parent, variable = variable, from_=LB, to=UB, resolution=STEP, orient='horizontal', length=250, command=command)
        #dummy_scale.configure(background='white')
        dummy_scale.pack()
        dummy_scale.pack(side=TOP,  fill="x")

    def create_GUI(self):

        #############################################
        ## All the controllers goes into the Listbox
        #############################################
        tabControl = ttk.Notebook(self.root, height=600, width=610)
        tab_overall = ttk.Frame(tabControl); tabControl.add(tab_overall, text ='Overall')
        tab_nose = ttk.Frame(tabControl); tabControl.add(tab_nose, text ='Nose')
        tab_eyebrows = ttk.Frame(tabControl); tabControl.add(tab_eyebrows, text ='Eyebrows')
        tab_eyes = ttk.Frame(tabControl); tabControl.add(tab_eyes, text ='Eyes')
        tab_lips = ttk.Frame(tabControl); tabControl.add(tab_lips, text ='Lips')
        tabControl.pack(side = 'left', padx = 30)

        ########################
        ## Overall Controllers
        ########################
        self.add_editing_scaler(scale_name = "Maximum Facial Width", parent = tab_overall, variable = self.maximum_facial_width, command = self.overall_editor_event)
        self.add_editing_scaler(scale_name = "Madibular Width", parent = tab_overall, variable = self.madibular_width, command = self.overall_editor_event)
        self.add_editing_scaler(scale_name = "Upper Facial Depth", parent = tab_overall, variable = self.upper_facial_depth, command = self.overall_editor_event)
        self.add_editing_scaler(scale_name = "Middle Facial Depth", parent = tab_overall, variable = self.middle_facial_depth, command = self.overall_editor_event)
        self.add_editing_scaler(scale_name = "Lower Facial Depth", parent = tab_overall, variable = self.lower_facial_depth, command = self.overall_editor_event)
        self.add_editing_scaler(scale_name = "Facial Height", parent = tab_overall, variable = self.facial_height, command = self.overall_editor_event)
        self.add_editing_scaler(scale_name = "Upper Facial Height", parent = tab_overall, variable = self.upper_facial_height, command = self.overall_editor_event)
        self.add_editing_scaler(scale_name = "Lower Facial Height", parent = tab_overall, variable = self.lower_facial_height, command = self.overall_editor_event)

        ########################
        ## Nose Controllers
        ########################
        self.add_editing_scaler(scale_name = "Nose Height", parent = tab_nose, variable = self.nose_height, command = self.nose_editor_event)
        self.add_editing_scaler(scale_name = "Nose width", parent = tab_nose, variable = self.nose_width, command = self.nose_editor_event)
        self.add_editing_scaler(scale_name = "Bridge Width", parent = tab_nose, variable = self.bridge_width, command = self.nose_editor_event)
        self.add_editing_scaler(scale_name = "Tip (+Up / -Down)", parent = tab_nose, variable = self.nose_tip_y, command = self.nose_editor_event)
        self.add_editing_scaler(scale_name = "Tip (+Out / -Out)", parent = tab_nose, variable = self.nose_tip_z, command = self.nose_editor_event)
        self.add_editing_scaler(scale_name = "Tip Width", parent = tab_nose, variable = self.tip_width, command = self.nose_editor_event)

        ########################
        ## Eyebrow Controllers
        ########################
        self.add_editing_scaler(scale_name = "Front Thickness", parent = tab_eyebrows, variable = self.front_thickness, command = self.eyebrows_editor_event)
        self.add_editing_scaler(scale_name = "Tail Thickness", parent = tab_eyebrows, variable = self.tail_thickness, command = self.eyebrows_editor_event)
        self.add_editing_scaler(scale_name = "Length", parent = tab_eyebrows, variable = self.eyebrow_length, command = self.eyebrows_editor_event)
        self.add_editing_scaler(scale_name = "Curve Strength", parent = tab_eyebrows, variable = self.curve_strength, command = self.eyebrows_editor_event)

        ########################
        ## Eyes Controllers
        ########################
        self.add_editing_scaler(scale_name = "Pupils Distance", parent = tab_eyes, variable = self.pupils_distance, command = self.eyes_editor_event)
        self.add_editing_scaler(scale_name = "Eye Height", parent = tab_eyes, variable = self.eye_height, command = self.eyes_editor_event)
        self.add_editing_scaler(scale_name = "Canthus Distance", parent = tab_eyes, variable = self.canthus_distance, command = self.eyes_editor_event)
        self.add_editing_scaler(scale_name = "Medial Canthus (+Up / -Down)", parent = tab_eyes, variable = self.medial_canthus_y, command = self.eyes_editor_event)
        self.add_editing_scaler(scale_name = "Lateral Canthus (+Up / -Down)", parent = tab_eyes, variable = self.lateral_canthus_y, command = self.eyes_editor_event)

        ########################
        ## Lip Controllers
        ########################
        self.add_editing_scaler(scale_name = "Labial Fissure Width", parent = tab_lips, variable = self.labial_fissure_width, command = self.ulip_editor_event)
        self.add_editing_scaler(scale_name = "Upper Lip Height", parent = tab_lips, variable = self.upper_lip_height, command = self.ulip_editor_event)
        self.add_editing_scaler(scale_name = "Upper Lip Width", parent = tab_lips, variable = self.upper_lip_width, command = self.ulip_editor_event)
        self.add_editing_scaler(scale_name = "Upper Lip-End Height", parent = tab_lips, variable = self.upper_lip_end_height, command = self.ulip_editor_event)
        self.add_editing_scaler(scale_name = "Lower Lip Height", parent = tab_lips, variable = self.lower_lip_height, command = self.llip_editor_event)
        self.add_editing_scaler(scale_name = "Lower Lip Width", parent = tab_lips, variable = self.lower_lip_width, command = self.llip_editor_event)
        self.add_editing_scaler(scale_name = "Lower Lip-End Height", parent = tab_lips, variable = self.lower_lip_end_height, command = self.llip_editor_event)



        # Rendered Image Display
        self.panel = tk.Label()
        self.panel.pack()
        self.panel.place(width=512, height=512, x=700, y=80)
        self.panel.configure(background='white')

        # Rotate
        labelRotate = tk.Label(self.root, text="Rotate")
        labelRotate.configure(font='bold', background='white')
        labelRotate.pack()
        labelRotate.place(height=LABEL_BOLD_H, width=100, x=680, y=570)
        self.scale_rotate = tk.Scale(self.root, variable = self.rotate_yaw_angle, from_=0, to=3.14/2, resolution=0.25, orient='horizontal', length=100, command=self.rotate_yaw_event)
        self.scale_rotate.pack()
        self.scale_rotate.configure(background='white')
        self.scale_rotate.place(x=680, y=600)

        # Open in Open3D Window Button
        self.btn_o3d = tk.Button(self.root)
        self.btn_o3d['text'] = "Open3D Viewer"
        self.btn_o3d['command'] = self.view_through_open3d
        self.btn_o3d.pack()
        self.btn_o3d.place(height=BUTTON_H*2, width=100, x=800, y=590)

        # Show Segments Button
        self.btn_seg = tk.Button(self.root)
        self.btn_seg['text'] = "Show Segments"
        self.btn_seg['command'] = self.show_segments_event
        self.btn_seg.pack()
        self.btn_seg.place(height=BUTTON_H*2, width=100, x=910, y=590)



    def rotate_yaw_event(self, val=None):
        """ Rotate scale event """
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)

    def display_image(self, rendered):
        """ Display the image in the image panel 
        rendered: the rendered image by o3d_render()
        """
        rendered = np.asarray(rendered)
        rendered = np.uint8(rendered * 255)
        img = ImageTk.PhotoImage(image=Image.fromarray(rendered))
        self.panel.configure(image = img)
        self.panel.image = img

    def render_image(self, V, T=None):
        # rotate
        V = self.V @ rotation_matrix(self.rotate_yaw_angle.get())
        # render
        mesh, rendered = o3d_render(V, T, Faces)
        return mesh, rendered

    def default_reconstruct(self):
        """ When the application start, use mean shape by default """
        part_latents = {}
        for key in latent_dims:
            part_latent = np.load('../saved_models/ffhq_mean_shape_latents/{}.npy'.format(key))
            part_latents[key] = torch.from_numpy(part_latent).to(device)
        V_overall, V_composed, part_latents, pred_offsets = reconstruction_from_latents(part_latents)
        self.part_latents = part_latents
        self.pred_offsets = pred_offsets
        V = ARAP_optimization(V_overall[0].detach().cpu().numpy(), V_composed[0].detach().cpu().numpy(), handle_ids)
        self.V = V
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)


    def view_through_open3d(self):
        """ Open an Open3D window """
        if self.mesh is not None and self.o3d_viewer_is_open == False:
            self.o3d_viewer_is_open = True
            o3d.visualization.draw_geometries([self.mesh])
            self.o3d_viewer_is_open = False
        else:
            print("Please load a face image first!")

    def show_segments_event(self):
        """ Show Segmentation Labels as Texture """
        if self.T is not None:
            self.T = None
            self.btn_seg['text'] = "Show Segments"
        else:
            self.T = seg_colors
            self.btn_seg['text'] = "Hide Segments"
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)


    ######################
    ## Controller Events
    def overall_editor_event(self, val=None):
        V = overall_editor( self.maximum_facial_width.get(),
                            self.madibular_width.get(),
                            self.upper_facial_depth.get(),
                            self.middle_facial_depth.get(),
                            self.lower_facial_depth.get(),
                            self.facial_height.get(),
                            self.upper_facial_height.get(),
                            self.lower_facial_height.get(),
                            self.part_latents['S_overall'],
                            self.V)
        self.V = V
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)

    def nose_editor_event(self, val=None):
        V = nose_editor(    self.nose_tip_y.get(), 
                            self.nose_tip_z.get(), 
                            self.nose_height.get(), 
                            self.nose_width.get(), 
                            self.tip_width.get(), 
                            self.bridge_width.get(),
                            self.part_latents['S_nose'],
                            self.pred_offsets,
                            self.V)
        self.V = V
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)

    def eyebrows_editor_event(self, val=None):
        V = eyebrows_editor(self.front_thickness.get(),
                            self.tail_thickness.get(),
                            self.eyebrow_length.get(),
                            self.curve_strength.get(),
                            self.part_latents['S_eyebrows'],
                            self.pred_offsets,
                            self.V)
        self.V = V
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)

    def eyes_editor_event(self, val=None):
        V = eyes_editor(    self.pupils_distance.get(),
                            self.eye_height.get(),
                            self.canthus_distance.get(),
                            self.medial_canthus_y.get(),
                            self.lateral_canthus_y.get(),
                            self.part_latents['S_eyes'],
                            self.pred_offsets,
                            self.V)
        self.V = V
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)

    def ulip_editor_event(self, val=None):
        V = ulip_editor(    self.labial_fissure_width.get(),
                            self.upper_lip_height.get(),
                            self.upper_lip_width.get(),
                            self.upper_lip_end_height.get(),
                            self.part_latents['S_ulip'],
                            self.pred_offsets,
                            self.V)
        self.V = V
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)

    def llip_editor_event(self, val=None):
        V = llip_editor(    self.lower_lip_height.get(),
                            self.lower_lip_width.get(),
                            self.lower_lip_end_height.get(),
                            self.part_latents['S_llip'],
                            self.pred_offsets,
                            self.V)
        self.V = V
        mesh, rendered = self.render_image(self.V, self.T)
        self.mesh = mesh
        # Display
        self.display_image(rendered)





## Start the GUI
root = tk.Tk(className="Face Reconstruction and Editing Demo")
root.configure(background='white')
root.geometry("1250x768")
app = Application(root=root)
app.mainloop()

# EOF