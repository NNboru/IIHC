from glob import glob
import cv2
import os
import shutil

if os.path.exists('Images_inpaint_resized/'):
    shutil.rmtree('Images_inpaint_resized/')
os.mkdir('Images_inpaint_resized/')

label = 0
WIDTH, HEIGHT = 256, 256
for folder in glob('Images_segmented/**/'):
    count=1
    fname = folder.split('\\')[1]
    print('processing - ',fname,end='..  ')
    samples = glob(folder+'Sample*/')
    os.mkdir('Images_inpaint_resized/'+str(label))
    for sample in samples:
        # 1st - inpainting image
        img1 = cv2.imread(sample + 'SegRGB1-2.jpg')
        mask1 = cv2.imread(sample + 'SegMask1-2.jpg', 0)
        img2 = cv2.inpaint(img1 ,mask1, 3 ,cv2.INPAINT_TELEA)

        # 2nd - resizing to (256*256) for VGG19
        img3 = cv2.resize(img2, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)

        # 3rd - storing images to "Images_inpaint_resized" folder
        img_path = 'Images_inpaint_resized/'+str(label)+'/'+str(count)+'.jpg'
        cv2.imwrite(img_path,img3)
        count+=1
    label+=1



