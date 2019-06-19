import os.path

org_kitti_path = "/SPACE/kan/Data/KITTI_raw_data/kitti/2011_09_26/"

# get all scenario folders
folderList = os.listdir(org_kitti_path)
folderList.sort()
sorted(folderList)

totalFrameNumber = 0
cloud_list = []

# count the total number of frames in all subdirectories
for currFolder in folderList:
    if '_sync' in currFolder :
        # get all frames in each  scenario folder
        fileList = os.listdir(org_kitti_path + currFolder + "/image_00/data")
        fileList.sort()
        sorted(fileList)
        totalFrameNumber += len(fileList)

        print ("scenario: " + str(currFolder) + "\texisting file number:\t" + str(len(fileList)) + "\tlast file name:\t" + str(fileList[-1]))

        for file in fileList:  # filter out all non jpgs

            if '.png' in file:
                cloud_list.append(org_kitti_path + currFolder + "/image_00/data/" + file)
            else:
                print ("\n ------ ERROR unknown file format: " + org_kitti_path + currFolder + '/' + file + "\n")

print (" input path: " + str(org_kitti_path))
print (" total Scenario: " + str(len(folderList)))
print (" total Frame: " + str(totalFrameNumber))
print (" total Cloud: " + str(len(cloud_list)))

