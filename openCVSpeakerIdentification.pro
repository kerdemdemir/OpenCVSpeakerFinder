TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG += c++14

QT       += core
QT       -= gui

INCLUDEPATH += "C:/range-v3-master/include/"
INCLUDEPATH += "$$_PRO_FILE_PWD_/includes/"
INCLUDEPATH += "D:/cvOutNoIPP/install/include"

QMAKE_CXXFLAGS += -std=gnu++1y -pthread -lpthread

LIB += -Wl,--stack,4194304

LIBS += -L"$$_PRO_FILE_PWD_/libs/" -lsndfile-1 -lws2_32 -lqjpeg4 -laubio-4 -lfftw3-3 -pthread
LIBS += -L"D:\cvOutNoIPP\bin" -lopencv_ml300  -lopencv_highgui300 -lopencv_features2d300 -lopencv_core300


SOURCES += main.cpp

HEADERS += \
    trainer.h \
    catch.hpp \
    pitchgrams.h \
    featureExtractor/featureExtractor.h \
    featureExtractor/featurelist.h \
    mlModel/pitchgrams.h \
    mlModel/modelbase.h \
    mlModel/gmmModel.h \
    statmanager.h \
    mlModel/tranierlist.h \
    featureExtractor/MFCCFeatures.h \
    general.h \
    featureExtractor/f0features.h \
    featureExtractor/f0highlevelfeatures.h


