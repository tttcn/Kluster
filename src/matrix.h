#ifndef MATRIX_H_
#define MATRIX_H_

/*
    matrix是处理的数据结构，数据存储在CPU上
*/

class Matrix
{
private:
shard_ptr<ELement> raw_ptr;
row_num;
col_num;
size;
public:
    Matrix();
    ~Matrix();
};



class Element{
    element_type;
    
}

class Vector{
   private:
    raw_ptr;
    length;
    size; 
}

#endif // MATRIX_H_