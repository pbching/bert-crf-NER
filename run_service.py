import os
import uvicorn

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
if __name__=='__main__':
    uvicorn.run('service:app', host='0.0.0.0', port=9883)