import numpy as np
import cv2 

def calculate_contour_coords(contour):
    approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)

    n = approx.ravel()
    i = 0
    
    for _ in n:
        if i % 2 == 0:
            x = n[i]
            y = n[i + 1]

            return (x, y)
        
def determine_staff(peoeple_frame):
    PEOPLE_SIZE = (74, 146)

    img = cv2.resize(peoeple_frame, PEOPLE_SIZE)
    people = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    people = cv2.GaussianBlur(people, (5, 5), 0)
    thresh = cv2.threshold(people, 100, 255, cv2.THRESH_BINARY)[1]

    # coordinates for the right segments
    top_left = (37, 15)
    bottom_right = (74, 103)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    MAX_AREA = 40
    dots = []
    for contour in cnts:
        area = cv2.contourArea(contour)
        if area < MAX_AREA:
            cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)
            dots.append(contour)
            (x, y) = calculate_contour_coords(contour)

            if (x > top_left[0] and y > top_left[1]) and (x < bottom_right[0] and y < bottom_right[1]):
                return 1

    else:
        return 0

# HOG descriptor to detect person in image
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("sample.mp4")
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detection starts here
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            # iterating to all the people detected
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # write the region of interest to a image file
            roi = frame[yA:yB, xA:xB]
            got_staff = determine_staff(roi)
            if got_staff == 1:
                # calculate staff's centroid coordinate
                staff_x = int((xA + xB) // 2)
                staff_y = int((yA + yB) // 2)

                print(f"Staff found at: X={staff_x} Y={staff_y}")

            # cv2.imwrite(f'people\\people_{count}.jpg', roi)
            count += 1

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break 

cap.release()
cv2.destroyAllWindows()
