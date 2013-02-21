TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    poisson.cpp

HEADERS += \
    data.h \
    common_inc.h \
    gui.hpp \
    poisson.h

win32 {
            INCLUDEPATH+="D:/Program Files/opencv/build/include/"
            INCLUDEPATH+="D:/Program Files/qwt/include"
            #QMAKE_LFLAGS += -static-libgcc -static-libstdc++
            LIBS+="D:/Program Files/opencv/build/x86/mingw/lib/libopencv_core242.dll.a"
            LIBS+="D:/Program Files/opencv/build/x86/mingw/lib/libopencv_highgui242.dll.a"
            LIBS+="D:/Program Files/opencv/build/x86/mingw/lib/libopencv_imgproc242.dll.a"
            DEPENDPATH+="D:/Program Files/opencv/build/x86/mingw/bin"
            DEPENDPATH+="D:/Program Files/qwt/lib/"
            DEPENDPATH+="D:/QtSDK/Desktop/Qt/4.8.1/mingw/bin"
}


unix {
            #INCLUDEPATH+=/usr/include/qwt
            LIBS+= -static-libgcc -static-libstdc++ -lGLU -lopencv_core -lopencv_highgui -lopencv_imgproc #-lqwt
}
