
import parameter
from tensorflow.python.tools import inspect_checkpoint as checkp

checkp.print_tensors_in_checkpoint_file(parameter.save_path,tensor_name='',all_tensors=True)
