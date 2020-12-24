import cv2
import numpy as np
import imutils.video
from timeit import time
from preprocess import number_edge, predict_light, crop_and_warp

from fuzzy_system.fuzzy_variable_output import FuzzyOutputVariable
from fuzzy_system.fuzzy_variable_input import FuzzyInputVariable
# from fuzzy_system.fuzzy_variable import FuzzyVariable
from fuzzy_system.fuzzy_system import FuzzySystem

density = FuzzyInputVariable('Density', 0, 15000, 100)
density.add_triangular('Sparse', -5000, 0, 6000)
density.add_triangular('Normal', 3500, 7000, 10000)
density.add_triangular('Crowd', 8000, 14000, 20000)
# density.add_triangular('Hot', 25, 40, 40)

# humidity = FuzzyInputVariable('Humidity', 20, 100, 100)
# humidity.add_triangular('Wet', 20, 20, 60)
# humidity.add_trapezoidal('Normal', 30, 50, 70, 90)
# humidity.add_triangular('Dry', 60, 100, 100)

light_time = FuzzyOutputVariable('Light_time', 0, 100, 100)
light_time.add_triangular('Short', 0, 0, 30)
light_time.add_triangular('Normal', 10, 30, 50)
light_time.add_triangular('Long', 40, 100, 100)
# motor_speed.add_triangular('Fast', 50, 100, 100)

system = FuzzySystem()
system.add_input_variable(density)
# system.add_input_variable(humidity)
system.add_output_variable(light_time)

system.add_rule(
		{ 'Density':'Sparse'},
		{ 'Light_time':'Long'})

system.add_rule(
		{'Density': 'Crowd'},
		{'Light_time': 'Short'})

system.add_rule(
		{'Density': 'Normal'},
		{'Light_time': 'Normal'})

# print(output)
# print('fuzzification\n-------------\n', info['fuzzification'])
# print('rules\n-----\n', info['rules'])

# system.plot_system()
def main():
    file_path = "ngatu_1.mp4"
    writeVideo_flag = True
    asyncVideo_flag = False
    if asyncVideo_flag:
            video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(w)
        print(h)
        out = cv2.VideoWriter("./result/" + file_path.split('.')[0] + ".avi", fourcc, 30, (w, h))
        frame_index = -1
    reference_frame = cv2.imread("./frame/ngatu_1_386.jpg")
    reference_edge = number_edge(reference_frame)
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    light_pts = np.array([(1015, 201), (1032, 209), (1017, 254), (998, 248)])
    density_red = []
    current_state = -1
    previous_state = -1
    a = 0
    b = []
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        t1 = time.time()
        print("1")
        light = crop_and_warp(frame, light_pts)
        index_max = predict_light(light)
        # frame_edge = number_edge(frame)
        # print(frame_edge - reference_edge)
        current_state = index_max
        if (index_max == 0):
            cv2.polylines(frame, [light_pts], True, (0, 0, 255), 3)
        elif (index_max == 2):
            cv2.polylines(frame, [light_pts], True, (0, 255, 0), 3)
            frame_edge = number_edge(frame)
            density_red.append(frame_edge - reference_edge)
        else:
            cv2.polylines(frame, [light_pts], True, (0, 255, 255), 3)

        if (previous_state == 0 and current_state != 0):
            print(np.mean(density_red))
            a = np.mean(density_red)
            # cv2.putText(frame, str(np.mean(density_red)), (300, 300), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX, 5, 3, cv2.LINE_AA)
            density_red.clear()
            print(density_red)
            output = system.evaluate_output({
                'Density': a
            })
            b = output
        cv2.putText(frame, str(a), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, str(b), (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        previous_state = current_state
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % (fps))
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))
if __name__ == '__main__':
    main()