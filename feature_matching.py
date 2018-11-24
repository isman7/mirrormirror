import cv2
import video

with video.VideoStream(0) as frames:
    ret, frame0 = next(frames)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame0, None)

    for ret, frame in frames:
        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(frame, None)

        # create BFMatcher object
        bf = cv2.BFMatcher()

        # Match descriptors.
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(frame0, kp1, frame, kp2, good, None, flags=2)

        cv2.imshow("features", img3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break