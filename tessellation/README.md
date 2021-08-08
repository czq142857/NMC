# tessellation
Tools for designing and visualizing cube tessellations.

## Requirements
- Python 3 with numpy and opencv-python.


## The pipeline

0. All scripts can be executed via *python xxx.py*.

1. Use *0_list_vertex_Fconnection_Cconnection_coloring_s4c2.py* to list all 37 unique cases with respect to rotation, mirroring, and inversion symmetry. The result configurations are saved in *LUT.npz*.

2. Use *1_create_all_configs.py* to create tessellation templates for all cases. The script will create a folder and write the templates for all cases into the folder. Then you can edit the templates to create your tessellations (see next section for details). A finished set of tessellation designs can be found in folder *configs_done*.

3. Use *2_count_connected_components.py* to count the number of connected components in each cube tessellation design from folder *configs_done*. This information is used for determining whether a cube in the ground truth mesh could be represented by the designed cube tessellations.

4. Use *3_verify_configs.py* to visually verify that the designed tessellations and the numbers of connected components are correct.

5. Use *4_upgrade_LUT_with_CC_info.py* to write *LUT_CC.npz* that contains the numbers of connected components.

6. Use *5_get_tessellation.py* to write the meshes in each cube into folder *output_mesh*. The normal direction for each triangle is computed automatically, and they should be verified by checking the meshes in folder *output_mesh*.

7. Use *6_get_LUT_with_tessellation.py* to get *LUT_tess.npz* that contains the cube tessellations. *LUT_tess.npz* is the only required file for running the NMC algorithm. Other files (*LUT.npz* and *LUT_CC.npz*) are only used in data pre-processing.


## The GUI

Run *template.py* or any script in *configs_done* to visualize the designed tessellations. These scripts are called templates. 

The first line in a template indicates the configuration of the cube, e.g.,```config_string = "101010001222220"```. It could be modified to represent any other cases. In the string, the first eight numbers indicate the signs of the eight cube vertices, then the next six numbers indicate the six face signs, and the last number is the tunnel flag. Number *2* means irrelevant (the sign does not affect the topology of the face or cube).

The third line in a template indicates the designed tessellation for the cube. The tessellation design is stored as a list of edges. ```load_connections = []``` means the template is blank, which is the default setting when a template is newly generated. Afterwards, you can run that template with python and modify the design, and the terminal will print out your current design as a string. You can copy that string and paste it into the third line to save/load your current design.

After opening the GUI by running a template, left-click and drag to rotate, right-click a dot and then drag to another dot to create an edge. The triangles are generated automatically according to the available edges. Note that in default, the GUI only shows the vertices that are usable in the specified cube case. To view all the vertices, set ```all_points_visibility[:] = 1``` between line 233 and 234 of the template.

You can modify *template.py* to add your customized functions. You can update all scripts in *configs_done* with your current *template.py* by running *update_template_to_configs.py*.



