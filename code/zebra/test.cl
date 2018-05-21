__kernel void gray( global const unsigned char *img_original, global unsigned char *img )
{
    img[ get_global_id(0) * 3 + 0 ] = img_original[ get_global_id(0) * 3 + 0 ];
    img[ get_global_id(0) * 3 + 1 ] = img_original[ get_global_id(0) * 3 + 0 ];
    img[ get_global_id(0) * 3 + 2 ] = img_original[ get_global_id(0) * 3 + 0 ];
}

__kernel void red( global const unsigned char *img_original, global unsigned char *img )
{
    img[ get_global_id(0) * 3 + 0 ] = 0.6 * img[ get_global_id(0) * 3 + 0 ];
    img[ get_global_id(0) * 3 + 1 ] = 0.5 * img[ get_global_id(0) * 3 + 1 ];
    img[ get_global_id(0) * 3 + 2 ] = 0.4 * img[ get_global_id(0) * 3 + 2 ];
}

#define TEST_INDEX(i,size) ((i >= 0 && i < size) ? true : false)
__kernel void kernel_test( global const unsigned char *img_original, global unsigned char *img, int array_size, int width, 
                           global const double *mask, int mask_size, double scaling_factor )
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

            //if( TEST_INDEX(index, array_size)) //who the fuck needs testing when you have the best bits
            {
                sum_r_x += mask[i_d + j_d * mask_size] * img_original[3 * index];
                sum_g_x += mask[i_d + j_d * mask_size] * img_original[3 * index + 1];
                sum_b_x += mask[i_d + j_d * mask_size] * img_original[3 * index + 2];
            }
        }
    }
    img[ 3 * (pos.x * width + pos.y) + 0 ] = (unsigned char)(sum_r_x / scaling_factor);
    img[ 3 * (pos.x * width + pos.y) + 1 ] = (unsigned char)(sum_g_x / scaling_factor);
    img[ 3 * (pos.x * width + pos.y) + 2 ] = (unsigned char)(sum_b_x / scaling_factor);
}