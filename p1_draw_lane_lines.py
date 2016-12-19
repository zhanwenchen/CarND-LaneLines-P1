def draw_lane_lines(img):

    # To use Canny Edge Detection, first convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use Gaussian smoothing to suppress noise and spurious
    # gradients by averaging
    # Define a kernel size for Gaussian smoothing / blurring
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # Canny edge detection on gray
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    ysize = image.shape[0]
    xsize = image.shape[1]
    apex_y = ysize/2+45
    vertices = np.array([[(0,ysize),(xsize/2-5, apex_y), (xsize/2+5, apex_y), (xsize,ysize)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 40
    max_line_gap = 30
    line_image = np.copy(image)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                                 min_line_length, max_line_gap)

    # Inelegantly averaging the slopes
    left_sum = [0,0,0,0]
    right_sum = [0,0,0,0]
    left_count = 0
    right_count = 0

    from operator import add

    # Iterate over the output "lines" and summing pairs of points
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1)) < 0:
                right_sum = map(add, right_sum, [x1, y1, x2, y2])
                right_count += 1
            else:
                left_sum = map(add, left_sum, [x1, y1, x2, y2])
                left_count += 1

    # Then get the average pair of points by dividing the sum
    left_line = [x // left_count for x in left_sum]
    left_slope = ((left_line[3]-left_line[1])/(left_line[2]-left_line[0]))
    left_const = left_line[1] - left_slope * left_line[0]
    right_line = [x // right_count for x in right_sum]
    right_slope = ((right_line[3]-right_line[1])/(right_line[2]-right_line[0]))
    right_const = right_line[1] - right_slope * right_line[0]

    # Then extrapolate the lines to the top and bottom in our
    # region of interest
    # y = ax + b, where we know y at top and bottom of region of interest
    left_top_y = apex_y
    left_top_x = (left_top_y - left_const) // left_slope
    left_bottom_y = ysize
    left_bottom_x = (left_bottom_y - left_const) // left_slope
    left_line = [int(left_top_x), int(left_top_y), int(left_bottom_x), int(left_bottom_y)]

    right_top_y = apex_y
    right_top_x = (right_top_y - right_const) // right_slope
    right_bottom_y = ysize
    right_bottom_x = (right_bottom_y - right_const) // right_slope
    right_line = [int(right_top_x), int(right_top_y), int(right_bottom_x), int(right_bottom_y)]

    draw_lines(line_image, [[left_line, right_line]],(255,0,0),10)

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 1, line_image, 1, 0)
    plt.imshow(lines_edges)
    return lines_edges

images_fname = os.listdir("test_images/")
for image_fname in images_fname:
    image = mpimg.imread("test_images/" + image_fname)
    image_with_lane_lines = draw_lane_lines(image)
    mpimg.imsave("test_images/after_"+image_fname, image_with_lane_lines)
