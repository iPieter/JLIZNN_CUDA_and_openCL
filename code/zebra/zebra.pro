#-------------------------------------------------
#
# Project created by QtCreator 2018-03-25T15:51:38
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = zebra
TEMPLATE = app

ICON = zebra.icns

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

linux-g++ | linux-g++-64 | linux-g++-32 {
    LIBS += -l OpenCL
}

unix {
    LIBS += -framework OpenCL
}

SOURCES += \
        main.cpp \
        mainwindow.cpp \
        pipeline.cpp

HEADERS += \
        mainwindow.h \
        std_image.h \
        std_image_write.h

FORMS += \
    mainwindow.ui

RESOURCES += \
    pipeline.qrc

