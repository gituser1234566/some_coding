#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
using namespace std;

class array_list{
private:    
    int *data;
   int capasity; 

public:


int size;

array_list(){

    size=0;
    capasity=1;
    data=new int[capasity];
    
}

array_list(vector<int>initial){

    size=0;
    capasity=1;
    data=new int[capasity];
    for(int e: initial){
        append(e);
    }
}


~array_list(){
    delete[] data;
}

void append(int n){

    if(size>=capasity){

        resize();
    }
    data[size]=n;
    size+=1;
}

void resize(){

    capasity*=2;
    int*tmp=new int[capasity];
    for(int i=0;i<size;i++){

        tmp[i]=data[i];
    }
    delete[] data;
    data=tmp;

}

int length(){
    return size;
}

void print(){

    std::cout<<"[";
    for(int i=0;i<size-1;i++){
        std::cout<<data[i];
        std::cout<<",";
    }
    std::cout<<data[size-1]<<"]"<<std::endl;
}

int &operator [](int i){
 if (0 <= i and i < size) {
        return data[i];
    } else {
        throw out_of_range("IndexError");
    }


}
void insert(int val, int index){
   if(index>size){
          throw out_of_range("IndexError");  
        }
    size+=1;
    
    int*tmp=new int[capasity+1];
    for(int i=0;i<=size;i++){

        
        tmp[i]=data[i];
        tmp[index]=val;
        if(i>=index){
        
        tmp[i]=data[i-1];
        
       
       }

       
    } 
    
    
    
    delete[] data;
    data=tmp;
    }

void remove(int index) {
    size-=1;
    int*tmp=new int[capasity];
    for(int i=0;i<=size+1;i++){
        tmp[i]=data[i];
        if(i>=index){
            
            tmp[i]=data[i+1];
        }
       
       } 
       if(size<0.25*capasity){
         shrink_to_fit();

       }
       
    

    delete[] data;
    data=tmp;

    } 



int pop(int index){
    int  rindex_vaue;
    size-=1;
    int*tmp=new int[capasity];
    for(int i=0;i<=size+1;i++){
        tmp[i]=data[i];
        if(i>=index){
            
            tmp[i]=data[i+1];
        }
       
       }

       rindex_vaue=data[index];
       
       if(size<0.25*capasity){
         shrink_to_fit();

       }
       delete[] data;
       data=tmp;

       return rindex_vaue; 

}  

int pop(){

   int  rindex_vaue;
    
    int*tmp=new int[capasity];
    for(int i=0;i<=size;i++){
        tmp[i]=data[i];
        
       
       }


       rindex_vaue=data[size-1];
       if(size<0.25*capasity){
         shrink_to_fit();

       }


       delete[] data;
       data=tmp;

       return rindex_vaue; 
 


 }




void shrink_to_fit(){
    
    
    int n;
    int n1;
    int n2;
    
    for(n=16;n>=1;n--){
        
        n1=pow(2,n);
        n2=pow(2,n-1);
      
        
        if(size-n1<size-n2){
            
            capasity=n1;
           if(size-n2>0){
               break;
           }

        }
        
        
        }
    
    
  


}

    
    
    
    



};


bool is_prime(int n){

    for(int i=2;i<=n;i++){
       if(n%i==0 && n!=i){
           return false;
       }

       else{
          
           continue;
           
          
       }
    }

return true;
    
    
}


void test_prime(){
array_list array;

int number;
number=0;
while(number<=10){

    for(int i=2;i<=40;i++){

        if(is_prime(i)==1){
            array.append(i);
            number+=1;
        }
        if(number==10){
         array.print();
         break;
        } 
    }
}
}

int main(){


vector<int> v = {2,3,5,7,11,13,17,19,23};

array_list primes (v);

int i,j;
i=0;
j=29;
 primes.insert(j,i);



cout<<primes.pop()<<endl;
cout<<primes[0]<<endl;
 test_prime();
 primes.print();







}





 