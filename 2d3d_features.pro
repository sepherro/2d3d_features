TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    2d3d_features.cpp

LIBS += -L"C:/opencv_qt/release/x64/mingw/lib"
LIBS += -L"C:/opencv_qt/debug/x64/mingw/lib"
CONFIG(debug, debug|release){
LIBS += -lopencv_highgui247d -lopencv_core247d -lopencv_imgproc247d -lopencv_calib3d247d -lopencv_features2d247d -lopencv_nonfree247d
} else {
LIBS += -lopencv_highgui247 -lopencv_core247 -lopencv_imgproc247 -lopencv_calib3d247 -lopencv_features2d247 -lopencv_nonfree247
}
INCLUDEPATH +=C:/opencv_qt/debug/include

HEADERS += \
    2d3d_features.hpp
