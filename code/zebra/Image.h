#ifndef IMAGE_H
#define IMAGE_H

#include <memory>
#include <string.h>

class Image
{
    private:

        int width;
        int height;
        int components;

        std::shared_ptr<unsigned char> img;

    public:
        
        Image( const std::string & file_name );

        int get_width() const;
        int get_height() const;
        int get_components() const;

        std::shared_ptr<unsigned char> get_img() const;
};

#endif