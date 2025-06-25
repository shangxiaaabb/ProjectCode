# C++ Learning
## 数组
> 所有代码：[⚙](./chapter04/)
**数组基本定义**： `typeNmae arrayName[arraySize]`。比如说下面几种定义方式都是合法的：

```c++
int tmp_array1[8]; // 直接定义 8个元素的数组
int tmp_array2[] = {0, 1, 2, 3}; // 不指定数组大小直接写入内容（内容数目就是数组大小）
int tmp_array2[10] = {0, 1, 2, 3};
```

**计算数组中元素个数**，可以直接通过：

```c++
sizeof tmp_array / sizeof tmp_array[0];
// 如果是字符数组
#include <cstring>
...
strlen(tmp_array);
```

`sizeof`:直接得到所占的字节大小；`strlen()`直接计算字符数组中的个数；


**向数组写入元素**，可以直接通过：

```c++
// 普通数组写入
int index = sizeof tmp_array / sizeof tmp_array[0]- 1; // 最后一个有效索引
cout << "Input new num: ";
cin >> tmp_array[index];
cout << "Updated num: " << tmp_array[index] <<endl;
// 面向行就行输入
cin.getline(tmp_array_3, sizeof tmp_array_3) 
```

> `cin.getline`支持3个参数：1、需要写入的**字符**数组；2、写入最大元素数量（会自动在字符串末尾添加 `\0`）；3、分隔符（遇到什么符号算终止 

**值得注意的是**：使用 `cin`写入数组时候，在键盘输入遇到空格时候，输入内容会被“截断”只有空格前的内容会被写入到数组中，**空格后的内容就会停留在缓存中**。比如说下面代码中，理论上要两次用户输入，但是实际只进行了一次用户输入：

```c++
cout<< "Enter your name:n";
cin>> name;
Cout << "Enter your favorite dessert:\n";
cin>> dessert;
cout<< "I have some delicious"<< dessert;
cout<< "for you" << name;
```

那么直接使用 `cin` 写入代码的逻辑就是：
![](https://s2.loli.net/2025/06/07/qmWzuH3gveMG5LZ.png)

那么直接使用 `cin.getline` 写入代码逻辑就是：
![](https://s2.loli.net/2025/06/07/gn8uqGw7Ddj9Lam.png)

值得注意的是，比如下面代码中使用 `cin >> tmp_array[index];` 而后输入回车就会导致在缓存区域里面存在 \n 因此通过 `cin.ignore();`（忽略（丢弃）输入流中的字符的函数） 进行去除。

### 字符串
c++中可以将字符串看作一种特殊的“数组”（也就意味着很多数组操作可以使用的操作在字符串中也可以使用，比如切片操作等），使用可以直接通过下面代码进行初始化：
```c++
#include <string>
using namespace std;
...
    string tmp_string0= "aabb";
    string tmp_string1 {"aabb"};
    string tmp_string2 = {"aabb"};
    ...
...
```
于此同时对于字符串而言支持直接对字符串进行相加操作（`tmp_string1+ tmp_string2`）；对于字符串其他操作：
```c++
#include <iostream>
#include <string>
#include <cstring>

using namespace std;
...
    char char1[100];
    string str1;
    string str2= "aabb";

    str1= str2; // 直接将 str2赋值给 str1

    strcpy(char1, str2);
    strcat(char1, str2);

    int len1= str2.size(); //计算长度
```

通过函数`strcpy( )`将字符串复制到字符数组中，使用函数`strcat( )`将字符串附加到字符数组末尾，通过`.size()`计算字符串个数。
> **值得注意**：字符串和字符数组之间还是存在差异的

### 结构
![](https://s2.loli.net/2025/06/08/zM8tj37FrDvfaUK.png)

在c++中可以使用结构来定义一个结构来存储多种数据类型的数据，使用和python中字典操作很像，比如说：
```c++
#include <iostream>

// 初始化一个结构
struct inflatable
{
    char name[20];
    float volume;
    double price;
};
...
    inflatable guest =
    {
        "Glorious Gloria",  // name value
        1.88,               // volume value
        29.99               // price value
    };
    cout << "Expand your guest list with " << guest.name;
    ...
...
```

最开始直接定义一个结构，并且结构里面的所存储的数据格式，使用时候只需要通过 `inflatable 结构名称`即可，并且访问结构也只需要：`结构名称.需要访问的结构成员`，除此之外对于 结构还可以使用类似数组的格式进行使用，比如说：
```c++
...
    inflatable peo1[1]={
            {"aa", 12, 11}
        }; // 其中 1 代表数量
    cout<<peo[1].name;
```