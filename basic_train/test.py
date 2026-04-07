num=0
for k in range(15,16):
    print("单比特旋转量：",k)
    for m in range(6,16):
        single_bit=1<<m
        print(k,m,single_bit,hex(single_bit))
        num+=1


# for k in range(1, 16):
#     print("双比特旋转量：",k)
#     for i in range(15):
#         for j in range(i+1,16):
#             double_bit=(1<<i)+(1<<j)
#             print(k,i,j,hex(double_bit))
#             num+=1
print(num)