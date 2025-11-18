from pioneer_sdk2 import Pioneer, Camera, CameraType, Event

main_camera = Camera(CameraType.MAIN)
opt_camera = Camera(CameraType.OPT)
pioneer = Pioneer()
pioneer.subscribe(lambda: print("Takeoff complete"), Event.TAKEOFF_COMPLETE)

while True:
    main_frame = main_camera.get_cv_frame()
    if main_frame is None:
        break
    opt_frame = opt_camera.get_cv_frame()
    if opt_frame is None:
        break
    
    print("left True") # left or right, is_stop