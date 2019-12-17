import os

DUTS_IMG_DIR = "/media/markytools/New Volume/Courses/EE298CompVis/finalproject/datasets/DUTS/DUTS-TR/DUTS-TR-Image/"
DUTS_INPAINT_DIR = "/media/markytools/New Volume/Courses/EE298CompVis/finalproject/datasets/DUTS/DUTS-TR/DUTS-TR-ANDAImg/"
DUTS_INPAINT_MASK = "/media/markytools/New Volume/Courses/EE298CompVis/finalproject/datasets/DUTS/DUTS-TR/DUTS-TR-ANDAMask/"

imgFileList = os.listdir(DUTS_IMG_DIR)
N_images = len(imgFileList)

inpaintList = os.listdir(DUTS_INPAINT_DIR)

def getFullInPaintFileName(partialName):
    for inpaintFName in inpaintList:
        if partialName in inpaintFName:
            return inpaintFName
    return None

### Real img file name coming from DUTS "ILSV...."
def getIndex(realImgFileName):
    print("realImgFileName")
    for i in range(0, N_images):
        # print("imgFileList[i]: ", imgFileList[i])
        if imgFileList[i] == realImgFileName:
            # print("Yes")
            return i
    return -1

for root, dirs, files in os.walk(DUTS_IMG_DIR):
        for file in files:
            if file.endswith('.jpg'):
                fullimgfilename = os.path.join(root, file)
                imgfilename = fullimgfilename.rsplit("/",1)[1]
                noextfilename = imgfilename[:-4]
                ind = getIndex(imgfilename)
                if ind != -1:
                    fullInpaintName = "image_" + str(ind).zfill(6)
                    fullFileInpaintNameFinal = getFullInPaintFileName(fullInpaintName)
                    fullMaskName = fullFileInpaintNameFinal.replace("image", "mask")
                    print("fullFileInpaintNameFinal: ", fullFileInpaintNameFinal)
                    print("fullMaskName: ", fullMaskName)
                    os.rename(DUTS_INPAINT_DIR+fullFileInpaintNameFinal, DUTS_INPAINT_DIR+noextfilename+".png")
                    os.rename(DUTS_INPAINT_MASK+fullMaskName, DUTS_INPAINT_MASK+noextfilename+".png")
