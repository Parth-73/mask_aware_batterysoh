"""Execute and persist the notebook, including partial outputs if a cell fails."""
from pathlib import Path
import asyncio
import nbformat
from nbclient import NotebookClient

if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
path=Path(__file__).resolve().parent/'CALCE_CNN_LSTM_Paper_Informed.ipynb'
nb=nbformat.read(path,4)
client=NotebookClient(nb,timeout=7200,kernel_name='isles-research',resources={'metadata':{'path':str(path.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Saved executed notebook',flush=True)
