"""Execute the notebook using the project interpreter, without a global kernel install."""
from pathlib import Path
import os,sys,json
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parent
jupyter=ROOT/'.jupyter';kernels=jupyter/'kernels';spec=kernels/'cs2-research';spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='CS2 research (.venv)',language='python')))
os.environ['JUPYTER_RUNTIME_DIR']=str(jupyter/'runtime');(jupyter/'runtime').mkdir(exist_ok=True)
os.environ['IPYTHONDIR']=str(jupyter/'ipython')
km=KernelManager(kernel_name='cs2-research',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
p=ROOT/'CALCE_CS2_Feature_Research.ipynb';nb=nbformat.read(p,as_version=4)
nb.metadata.kernelspec=dict(display_name='CS2 research (.venv)',language='python',name='cs2-research')
client=NotebookClient(nb,km=km,timeout=3600,resources={'metadata':{'path':str(ROOT)}})
try:client.execute()
finally:
 if km.has_kernel:km.shutdown_kernel(now=True)
 nbformat.write(nb,p)
print('Executed notebook saved:',p)
