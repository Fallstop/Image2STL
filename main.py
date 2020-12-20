import argparse
from threading import Thread, Event, Lock
import multiprocessing
import traceback

import numpy as np
from numpy.core.fromnumeric import transpose
from numpy.lib.function_base import average
from stl import mesh, Mode
from PIL import Image
import trimesh
import progressbar

# So that the program output doesn't interfer with the progress bar
import logging
progressbar.streams.wrap_stderr()
logging.basicConfig(level=logging.DEBUG)

data_lock = Lock()
event = Event()

print(mesh.Mesh.dtype)

Image.MAX_IMAGE_PIXELS = None

MULTITHREADING_THRESHOLD = 1000

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a height map into an STL model.')
    parser.add_argument('inputImage', type=str, help='The input heighmap path.')
    parser.add_argument('outputSTL', help='The new putput STL file path')
    parser.add_argument('-n','--nullData', dest='nullData', help='The null data value in the image', default=-32768)
    return parser.parse_args()

def get_heights(inputImage):
    
    heightData = np.zeros(inputImage.shape[0:2])
    for x,row in enumerate(inputImage):
        for y,rgb in enumerate(row):
            heightData[x][y] = np.average(rgb)
    print("Yes: ",heightData.shape)
    return heightData

def generate_mesh_thread(heightData,yOffset,ID, args):
    global generatedTriangles
    global threadProgress
    try:
        dataLength = (len(heightData)-1)*(len(heightData[0])-1)
        triangleMesh = np.zeros(dataLength, dtype=mesh.Mesh.dtype)

        for x,row in enumerate(heightData[:-1]):
            for y,height in enumerate(row[:-1]):
                if height == args.nullData:
                    continue
                
                if x%2 == 0:
                    triangle = np.array([
                        [x, y, height],
                        [x, y+1, heightData[x][y+1]],
                        [x+1, y+1, heightData[x+1][y+1]]
                        ])
                    
                else:
                    triangle = np.array([
                        [x, y, height],
                        [x-1, y, heightData[x-1][y]],
                        [x-1, y+1, heightData[x-1][y+1]]
                    ])
                triangleMesh["vectors"][x*y] = triangle
            if x%3==0 and ID >= 0:
                threadProgress[id] = x*y
                if event.is_set():
                    break
        if ID >= 0:
            logging.info("Thread",i,"is finished")
            with data_lock:
                generatedTriangles['vectors'][:,yOffset:len(triangleMesh)] = triangleMesh
        else:
            return triangleMesh
    except Exception as e:
        logging.error("Error in thread {}! Error:\n{}".format(ID,e))
        traceback.print_exc()
    

def main(args):
    global generatedTriangles
    global threadProgress

    logging.info("Reading Image")
    inputImage = np.array(Image.open(args.inputImage))
    

    logging.info("Extracting Heights")
    heightData = get_heights(inputImage)
    

    logging.info("Generating triangles from data")
    dataLength = (len(heightData)-1)*(len(heightData[0])-1)
    generatedTriangles = np.zeros(dataLength, dtype=mesh.Mesh.dtype)
    logging.debug("Image is {} tall and {} wide with an estimated {} triangles".format(len(heightData),len(heightData[0]),dataLength))


    #Decide if threading is required
    if len(heightData) > MULTITHREADING_THRESHOLD:
        logging.info("Using multithreading")

        np.zeros(dataLength, dtype=mesh.Mesh.dtype)

        #Spawn threads
        threadCount = multiprocessing.cpu_count()
        dataSectionSize, dataSectionExtra = divmod(len(heightData)-1, threadCount)
        dataSections = [0] + [dataSectionSize*(index+1) for index in range(threadCount-1)] + [len(heightData)-dataSectionExtra-1]
        threads = []

        threadProgress = np.zeros((threadCount,))
        logging.debug("Threads Started with sections {}".format(dataSections))
        for i in range(threadCount):
            threads.append(Thread(target=generate_mesh_thread, args=(heightData[dataSections[i]:dataSections[i+1]],dataSections[i],i,args)))
        
        with progressbar.ProgressBar(max_value=dataLength) as bar:
            while True:
                try:
                    if any(thread.is_alive() for thread in threads):
                        bar.update(np.average(threadProgress))
                    else:
                        break
                except KeyboardInterrupt:
                    event.set()
                    break
    else:
        logging.info("Using single threading due to small image size")
        generatedTriangles = generate_mesh_thread(heightData,0,-1,args)
        print(generatedTriangles)
    with open("test.txt", mode="w") as f:
        f.write(np.array2string(generatedTriangles,threshold=10000000000))
        f.close()

    logging.info("Generating Mesh")
    generatedMesh = mesh.Mesh(generatedTriangles)

    logging.info("Save STL")
    generatedMesh.save(args.outputSTL, mode=Mode.ASCII)
    
    


if __name__ == "__main__":
    args = parse_args()
    main(args)