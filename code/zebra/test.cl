__kernel void gray( global const unsigned char *img_original, global unsigned char *img, int width, int comp )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};

    img[ comp * (pos.x * width + pos.y) + 0 ] = img[ comp * (pos.x * width + pos.y) + 0 ];
    img[ comp * (pos.x * width + pos.y) + 1 ] = img[ comp * (pos.x * width + pos.y) + 0 ];
    img[ comp * (pos.x * width + pos.y) + 2 ] = img[ comp * (pos.x * width + pos.y) + 0 ];
}

__kernel void sync_images( global unsigned char *img_original, global unsigned char *img, int width, int comp )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};

    for (int i = 0; i < comp; i++) 
        img_original[ comp * (pos.x * width + pos.y) + i ] = 0;
}

__kernel void image_grading( global const unsigned char *img_original, global unsigned char *img, int width, int comp, int r, int g, int b)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};

    img[ comp * (pos.x * width + pos.y) + 0 ] = clamp( (int) 255.0 * img[ comp * (pos.x * width + pos.y) + 0 ] / r, 0, 255 );
    img[ comp * (pos.x * width + pos.y) + 1 ] = clamp( (int) 255.0 * img[ comp * (pos.x * width + pos.y) + 1 ] / g, 0, 255 );
    img[ comp * (pos.x * width + pos.y) + 2 ] = clamp( (int) 255.0 * img[ comp * (pos.x * width + pos.y) + 2 ] / b, 0, 255 );

}

#define TEST_INDEX(i,size) ((i >= 0 && i < size) ? true : false)
__kernel void kernel_test( global const unsigned char *img_original, global unsigned char *img, int array_size, int width, 
                           global const double *mask, int mask_size, double scaling_factor, int comp )
{
    const int2 pos = {get_global_id(0), get_global_id(1)};

    double sum_r_x = 0;
    double sum_g_x = 0;
    double sum_b_x = 0;

    for( int i_d = 0; i_d < mask_size; i_d++ )
    {
        int i = i_d - mask_size / 2;
        for( int j_d = 0; j_d < mask_size; j_d++ )
        {
            int j = j_d - mask_size / 2;
            int index = ((pos.x + i) * width + pos.y + j);

            if( TEST_INDEX(index, array_size)) //who the fuck needs testing when you have the best bits
            {
                sum_r_x += mask[i_d + j_d * mask_size] * img_original[ comp * index];
                sum_g_x += mask[i_d + j_d * mask_size] * img_original[ comp * index + 1];
                sum_b_x += mask[i_d + j_d * mask_size] * img_original[ comp * index + 2];
            }
        }
    }
    img[ comp * (pos.x * width + pos.y) + 0 ] = (unsigned char)(sum_r_x / scaling_factor);
    img[ comp * (pos.x * width + pos.y) + 1 ] = (unsigned char)(sum_g_x / scaling_factor);
    img[ comp * (pos.x * width + pos.y) + 2 ] = (unsigned char)(sum_b_x / scaling_factor);
}