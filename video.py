import cv2


class VideoStream(object):
    def __init__(self, input_video):
        self.input_video = input_video
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.input_video)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __next__(self):
        return self.cap.read()

    def __iter__(self):
        return self

    def release(self):
        self.cap.release()


if __name__ == '__main__':

    with VideoStream(0) as video_stream:
        for ret, frame in video_stream:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

