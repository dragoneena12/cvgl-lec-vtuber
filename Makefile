CXX=g++
CXXFLAGS=`pkg-config opencv --cflags`
LDLIBS=-framework ApplicationServices `pkg-config opencv --libs`