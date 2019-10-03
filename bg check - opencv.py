from PIL import Image
import cv2 # equivalent Java library is at https://opencv-java-tutorials.readthedocs.io/en/latest/
import numpy as np

RUN_CONFIG = 1          # select input fileset
DEBUGGING = 0

REKOG_BBOX_WIDTH = 60   # for testing, assume ~1/3 image width
BBOX_TOLERANCE = 1.33   # testing example value

TOLERANCE = 30          # variance smoothness upper limit

FILE_METADATA = [
    {
        'filename' : 'photos2.bmp',
        'width' : 140,
        'height' : 140 } ,
    {
        'filename' : 'jason.bmp',
        'width' : 150,
        'height' : 200 }
]
                      
# setup constants
f = FILE_METADATA[RUN_CONFIG]
filename = f['filename']
width = f['width']
height = f['height']

# load and flatten bmp
master_img = Image.open(filename).convert('L')

# crop 3x3 composite image into a list of smaller ones
img = []
for i,y in enumerate((0,height,2*height)):
    im = []
    for j,x in enumerate((0,width,2*width)):
        im.append(master_img.crop((x,y,x+width-1,y+height-1)))
        if DEBUGGING: im[j].show()
    img.append(im)
    
# loop over individual images
for i in range(3):
    for j in range(3):

        objects_found = False
        print("\nProcessing image ",i,j)

# extract pixel values
        pixels = list(img[i][j].getdata())

# original algorithm - calculate variance of pixels on L & R sides of head
# bounding area is set manually from inspection of sample images

        sx2, sx = 0,0
        ix = 0
        sidebars = int(width*0.25) + int(width*0.25)%2
        N = sidebars * int(height//2)
        for y in range(int(height//2)):
            for x in set(range(sidebars//2)).union(set(range(width-sidebars//2,width))):
                p = pixels[x+(width-1)*y]
                sx2 += p**2
                sx += p
                ix+=1
                if DEBUGGING and not ix%50:
                    print(ix,N,p,int(sx/ix),int(sx2/ix-(sx/ix)**2),sep='\t')
        varM = sx2/N - (sx/N)**2

# alternative algorithm using edge detection to remove head
# convert image to cv2 format and crop bottom half off

        imcv_full = np.asarray(img[i][j])
        imcv = imcv_full[0:height//2-1][0:width-1]
        if DEBUGGING:
            cv2.imshow("source",imcv)
            cv2.waitKey(0)

# perform edge detection (blur to soften then Canny algorithm)
        blurred = cv2.blur(imcv, (3,3))
        canny = cv2.Canny(blurred, 50, 200)
        if DEBUGGING:
            cv2.imshow("edges",canny)
            cv2.waitKey(0)

# find bbox of edges - add small safety border
        pts = np.argwhere(canny>0)
        y1,x1 = pts.min(axis=0)
        y2,x2 = pts.max(axis=0)
        x1 = x1 if x1<3 else x1-2
        y1 = y1 if y1<3 else y1-2
        x2 = x2 if x2>136 else x2+2
        y2 = height//2 - 2 # always go to bottom of crop
        
# compare bbox to AWS Rekognition bbox
# if much bigger then b/g objects are present or strong b/g texture

        if (x2-x1) > REKOG_BBOX_WIDTH * BBOX_TOLERANCE:
            objects_found = True

# show the crop-out area
        if DEBUGGING:
            demo = imcv.copy()
            cv2.rectangle(demo, (x1, y1), (x2, y2), 0, 1)
            cv2.imshow("bbox",demo)

# finally get variance of pixels outside the crop-out area
        sx2, sx = 0,0
        ix = 0
        # num pixels we will scan
        N = (height//2-1)*(width-1) - (x2-x1+1)*(y2-y1+1)
        for y in range(height//2-1):
            for x in range(width-1):
                if not (x<=x2 and x>=x1 and y<=y2 and y>=y1):
                    p = imcv[y][x]
                    sx2 += p**2
                    sx += p
                    ix+=1
                    if DEBUGGING and not ix%50:
                        print(ix,N,p,int(sx/ix),int(sx2/ix-(sx/ix)**2),sep='\t')
        varA = sx2/N - (sx/N)**2
        
# results
        print("Variance (crop) ",varM)
        print("Variance (edge) ",varA)

        if objects_found:    
            print("REJECT: objects in background")
        elif varA > TOLERANCE:
            print("REJECT: background not plain / shadow")
        else:
            print("ACCEPT")

        if DEBUGGING: cv2.waitKey(0)

if DEBUGGING:
    print("Any key to exit")
    cv2.waitKey(0)

cv2.destroyAllWindows()

        
