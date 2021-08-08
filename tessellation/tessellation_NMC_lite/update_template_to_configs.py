import os

done_list = [name for name in os.listdir("configs_done") if name[-3:]==".py"]

template_py = open("template.py", 'r')
template_py_lines = template_py.readlines()
template_py.close()

for config_name in done_list:
	config_py_name = "configs_done/"+config_name
	config_py = open(config_py_name, 'r')
	config_py_lines = config_py.readlines()
	config_py.close()
	
	config_py = open(config_py_name, 'w')
	for j in range(len(template_py_lines)):
		if j<4:
			config_py.write(config_py_lines[j])
		else:
			config_py.write(template_py_lines[j])
	config_py.close()