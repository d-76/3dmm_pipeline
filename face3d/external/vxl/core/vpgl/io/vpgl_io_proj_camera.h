#ifndef vpgl_io_proj_camera_h_
#define vpgl_io_proj_camera_h_
//:
// \file
#include <vsl/vsl_binary_io.h>
#include <vpgl/vpgl_proj_camera.h>

//: Binary save camera to stream
template <class T>
void vsl_b_write(vsl_b_ostream & os, vpgl_proj_camera<T> const& camera);

//: Binary load camera from stream.
template <class T>
void vsl_b_read(vsl_b_istream & is, vpgl_proj_camera<T> &camera);

//: Print human readable summary of object to a stream
template <class T>
void vsl_print_summary(std::ostream& os,const vpgl_proj_camera<T> & c);

#endif
