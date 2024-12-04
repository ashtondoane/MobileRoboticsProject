"""
 * ThymioVision Class Definition
 *
 * For a thorough report of this class, please consider the file report_cv.ipynb
 * 
 * @author Ashton Doane
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ThymioVision:
    @staticmethod
    def calibrateCameraPos(camNum):
        """
        Position the camera such that it aligns with the corners of A0 paper as shown. This
        is purely for user setup, and does not return a value. If dots are aligned with the corners of
        A0 paper, ensures that 1 px = 0.9344 mm
        """
        cv2.namedWindow("Camera Calibration") 
        vc = cv2.VideoCapture(camNum)
        ret = True
        while True:
            ret, frame = vc.read()
            if not ret:
                break
            
            # add calibration circles to frame
            cv2.circle(frame, (360, 90), 5, (0, 0, 255), 5)
            cv2.circle(frame, (frame.shape[1]-360, 90), 5, (0, 0, 255), 5)
            cv2.circle(frame, (360, frame.shape[0]-90), 5, (0, 0, 255), 5)
            cv2.circle(frame, (frame.shape[1]-360, frame.shape[0]-90), 5, (0, 0, 255), 5)

            cv2.imshow("Camera Calibration", frame)
            key = cv2.waitKey(50)
            if key == ord('c'): # Escape and return image on c
                break
        
        vc.release()
        cv2.destroyAllWindows()

    @staticmethod
    def captureImageOnCommand(camNum):
        """
        Provides the user with a camera feed, from which the user may input 'C' to
        capture the image provided. Does not complete without user input.
        @param cv2 BGR image, from which we extract edges.
        @returns cv2 grayscale image with detected edges from input img.
        """
        cv2.namedWindow("Camera View")
        vc = cv2.VideoCapture(camNum)
        ret = True
        while True:
            ret, frame = vc.read()
            if not ret:
                break
            cv2.imshow("Camera View", frame)
            key = cv2.waitKey(50)
            if key == ord('c'): # Escape and return image on c
                break
        
        vc.release()
        cv2.destroyAllWindows()
        return frame

    # def captureImage():
    #     vc = cv2.VideoCapture(0)
    #     ret, frame = vc.read()
    #     vc.release()
    #     return frame

    @staticmethod
    def getEdges(img, filter = 'median', edge_method = 'canny', verbose=False):
        """
        Extract detected edges from a provided image.
        @param img(cv2 BGR_image): Image from which we extract edges.
        @param filter (string): Indication of what type of filter to overlay on the image.
        @param edge_method (string): Indication of what type of edge detection method should be used.
        @param verbose (bool): If true, will display each step of the processing.
        @returns cv2 grayscale image with detected edges from input img.
        """
        # First, convert the input image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply the selected filter
        if filter == 'median':
            filtered_img =  cv2.medianBlur(gray_img, 31)
        elif filter == 'average':
            pass
        elif filter == 'gaussian':
            pass
        else:
            filtered_img = gray_img

        #Apply the selected edge detection method to the filtered image
        if edge_method == 'canny':
            edges = cv2.Canny(filtered_img, 100,200)
        else:
            pass

        # If verbose selected, Display images
        if verbose:
            # Set up plot size
            plt.rcParams["figure.figsize"] = (20,5)
            plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.4)
            
            #Grayscaled Image:
            plt.subplot(1, 3, 1)
            plt.title("Grayscale")
            plt.imshow(gray_img, cmap='gray')

            #Filtered Image:
            plt.subplot(1, 3, 2)
            plt.title("Filtered: " + filter)
            plt.imshow(filtered_img, cmap='gray')

            #Edges + expansion radius Image:
            plt.subplot(1, 3, 3)
            plt.title("Edges: " + edge_method)
            plt.imshow(edges, cmap='gray')

            plt.show()
            
        return edges

    @staticmethod
    def pixelToRealSpace(position):
        """
        Converts a pixel location to a dimension in real space. Coordinate frame centered on the top left corner of the paper.
        As the setup always ensures alignment of the camera to the corners of A0 paper, the ratio is set.
        @param position (x,y): Pixel location on the camera image.
        @returns (x,y) tuple of location in real space in cm.
        """
        # Camera shape (1080, 1920, 3)
        # Size of bounding box (900, 1200)
        # Paper dimensions (675 x 900 mm)
        # Alignment from calibration such that 1 px = 0.75 mm
        return ((position[0]-360)*0.75/10, (position[1]-90)*0.75/10)
    
    @staticmethod #66 cm x 90 --> 
    def realSpaceToPixel(position):
        """
        Converts a real location to a location on the camera. Coordinate frame centered on the top left corner of the paper.
        As the setup always ensures alignment of the camera to the corners of A0 paper, the ratio is set.
        @param position (x,y): Pixel location on the camera image.
        @returns (x,y) tuple of location in real space in cm.
        """
        # Camera shape (1080, 1920, 3)
        # Paper dimensions (841 x 1189mm)
        # Alignment from calibration such that 1 px = 0.9344 mm
        return (int(position[0]*10/0.75)+360, int(position[1]*10/0.75)+90)

    #TODO: ADD VISUALIZATION OF STEPS
    @staticmethod
    def detectBlueDot(frame, divisions=1, minScale=1, maxScale = 2, method = 'TM_CCORR_NORMED', templatePath = "Templates/blueDot.png", verbose=False):
        """
        Note: Does NOT support TM_SQDIFF or SQDIFF_NORMED
        """
        template = cv2.imread(templatePath) #read template as bgr image
        globalMax = 0
        best_approx = ([], 0, 0, 0) #pos/w/h/scale
        
        # resize the template image to a variety of scales, perform matching 
        for scale in np.linspace(minScale, maxScale, divisions)[::-1]:
            resized = cv2.resize(frame, (0,0), fx=scale, fy=scale) #resize
            
            # get effective size of rectangle bounding box we are searching
            w, h, c = template.shape
            w = int(w/scale)
            h = int(h/scale)

            meth = getattr(cv2, method)

            # Apply template Matching
            res = cv2.matchTemplate(resized,template,meth)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
            if  max_val > globalMax:
                globalMax = max_val
                best_approx = ([int(max_loc[0]/scale), int(max_loc[1]/scale)], w, h, scale)


        top_left,w,h,scale = best_approx
        bottom_right = (top_left[0] + w, top_left[1] + h)

        if verbose:
            copy = frame.copy()
            plt.rcParams["figure.figsize"] = (16,4)
            cv2.rectangle(copy, top_left, bottom_right, (255, 50, 255), 5)
            plt.imshow(copy,cmap = 'gray')
            plt.show()
        
        x = top_left[0] + int(w/2)
        y = top_left[1] + int(h/2)
        return (x,y) #return center of box
    
    @staticmethod
    def detectGoal(frame, divisions=4, method = 'TM_CCORR_NORMED', templatePath = "Templates/greenDot2.png", verbose=False):
        """
        Note: Does NOT support TM_SQDIFF or SQDIFF_NORMED
        """
        template = cv2.imread(templatePath) #read template as bgr image
        globalMax = 0
        best_approx = ([], 0, 0, 0) #pos/w/h/scale
        
        # resize the template image to a variety of scales, perform matching 
        for scale in np.linspace(0.5, 2, divisions)[::-1]:
            resized = cv2.resize(frame, (0,0), fx=scale, fy=scale) #resize copy
            
            # get effective size of rectangle bounding box we are searching
            w, h, c = template.shape
            w = int(w/scale)
            h = int(h/scale)

            meth = getattr(cv2, method)

            # Apply template Matching
            res = cv2.matchTemplate(resized,template,meth)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
            if  max_val > globalMax:
                globalMax = max_val
                best_approx = ([int(max_loc[0]/scale), int(max_loc[1]/scale)], w, h, scale)


        top_left,w,h,scale = best_approx
        bottom_right = (top_left[0] + w, top_left[1] + h)

        if verbose:
            copy = frame.copy()
            plt.rcParams["figure.figsize"] = (16,4)
            cv2.rectangle(copy, top_left, bottom_right, (255, 50, 255), 5)
            plt.imshow(copy,cmap = 'gray')
            plt.show()
        
        x = top_left[0] + int(w/2)
        y = top_left[1] + int(h/2)
        return (x,y) #return center of box
    
    @staticmethod
    # TODO BASE ON MEDIAN RATHER THAN AVERAGE
    def detectOrangeHeading(frame, reduction = 0.3, THRESHOLD = 25, filter_size = 10, verbose = False):
        filtered =  cv2.blur(frame, (filter_size,filter_size))
        lower_quality = cv2.resize(filtered, (0,0), fx = reduction, fy = reduction) # rescale for faster processing
        # Create a mask that looks for only the light indicator for position
        hsv = cv2.cvtColor(lower_quality, cv2.COLOR_BGR2HSV) #convert to hsv for masking
        lower_orange = np.array([0, 20, 220]) # hue/saturation/brightness
        upper_orange = np.array([60, 255, 255]) 
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        result = cv2.bitwise_and(lower_quality, lower_quality, mask=mask) # image correcting

        centerX = 0
        centerY = 0
        numDataPoints = 0
        for i, row in enumerate(result):
            for j, pixel in enumerate(row):
                if pixel.any() != 0:
                    centerX += j
                    centerY += i
                    numDataPoints += 1
        if numDataPoints < THRESHOLD:
            if verbose: 
                print("Orange not found")
            return (None, None)
        centerX = int(centerX/numDataPoints/reduction) # find average and rescale to full value
        centerY = int(centerY/numDataPoints/reduction)
        
        # If verbose, display each step of processing
        if verbose:
            plt.subplot(121), plt.imshow(result)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.subplot(122), plt.imshow(rgb)
            plt.plot([centerX], [centerY], 'o')
            plt.show()
        pass

        return (centerX, centerY)
                
                
            
    @staticmethod
    def getThymioPose(frame, verbose=False):
        """
        Extracts the Thymio pose from a camera feed and returns as a triple of (x,y,theta), relative to the top-left corner of the camera.
        @param frame (np.array): BGR cv2 image to extract position from.
        @returns (x, y, theta, size)
        """
        relevantFrame = frame[90:-90, 360:-360]
        blueX, blueY = ThymioVision.detectBlueDot(relevantFrame, minScale=0.5, maxScale=0.5, divisions=1)
        orangeX, orangeY = ThymioVision.detectOrangeHeading(relevantFrame, reduction=0.3, THRESHOLD=15, filter_size=10)

        if blueX is None or orangeX is None:
            print('Thymio not found.')
            return (None, None, None)
        
        blueX += 360
        blueY += 90
        orangeX += 360
        orangeY += 90

        dx = float(orangeX-blueX)
        dy = -float(orangeY-blueY)
        theta = np.arctan2(dy,dx)

        if np.sqrt(dx**2 + dy**2 ) > 300:
            print('Thymio not found. Distance too large')
            return (None, None, None)

        if verbose:
            plt.rcParams["figure.figsize"] = (16,4)
            plt.imshow(frame)
            plt.quiver(blueX, blueY, dx, dy, color='red')
            plt.show()
        return (blueX, blueY, theta)


    @staticmethod 
    def getMap(frame, verbose=False):
        """
        Determine the map of the layout by considering thymio position, size, detected edges, and goal position. 
        @param frame (np.array): A camera image
        @returns Tuple (map, start, goal) with types (np.array, [x,y], [x,y]) representing the map of edges, start location
        and goal position for the A* algorithm.
        """
        # Get edge list
        edges = ThymioVision.getEdges(frame)

        # Find start and goal position
        startPos = ThymioVision.getThymioPose(frame)[0:2]
        if not startPos:
            print("Thymio start position not found")
        tSize =  75 # based on the calibrated camera, the thymio can be approximated by this radius
        goalPos = ThymioVision.detectGoal(frame)
        if not goalPos:
            print("Goal not found")
            return
        
        if startPos[0] is None:
            print("Invalid map. Thymio not found.")
            return
        if goalPos[0] is None:
            print("Invalid map. Goal not found.")
            return
        # clear out space around goal and start
        cv2.circle(edges, startPos, radius=int(tSize), thickness=-1, color=0)
        cv2.circle(edges, goalPos, radius=int(tSize), thickness=-1, color=0)
        #radius
        final_map = np.zeros(shape=edges.shape)
        for i, row in enumerate(edges):
            for j, pixel in enumerate(row):
                if pixel == 255:
                    cv2.circle(final_map, (j,i), radius=int(tSize), thickness=-1, color=1)

        if verbose:
            plt.plot(startPos[0], startPos[1], 'o', color='red')
            plt.plot(goalPos[0], goalPos[1], 'o', color='green')
            plt.imshow(final_map)
        return (final_map, startPos, goalPos)
