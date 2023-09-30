#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <stdlib.h>
using namespace std;




struct Node{
int val;
Node*next;
Node*prev;

Node(int n) {
    val = n;
    next = nullptr;
}

Node(int n, Node* p) {
    val = n;
    next = p;
}

};






class CircularLinkedList{
private:

Node*head;
Node*tail;
int count;

public:

CircularLinkedList(vector<int>initial){

count=0;
head=nullptr;
tail=nullptr;


  for(int e: initial){
        append(e);
    }

}

CircularLinkedList(int n){
count=0;
head=nullptr;
tail=nullptr;   
for(int i=1;i<=n;i++){
        append(i);
    }

}

CircularLinkedList(){

count=0;
head=nullptr;
tail=nullptr;


  

}


~CircularLinkedList(){
  Node* current;
    Node* next;
    
    current = head;
    
    
    

     while (current != head) {
        next = current->next;
        delete current;
        current = next;
    }

    delete head;

    

}

void append(int val) {
    count++;
    if (head == nullptr) {
        head= new Node(val);
        tail=head;
        return ;
        }

    tail->next= new Node(val);
    tail=tail->next;
    
    tail->next=head;
    

    }   

Node* get_node(int index) {
    if (length()==0) {
        
        throw out_of_range("list is empty");
    }
    

    
    Node* current = head;
    for (int i=0; i<index; i++) {
        current = current->next;
        
    }
    return current;
}

int& operator[](int index) {
    return get_node(index)->val;
}

void print() {
    Node* current = head;
    
    
    cout << "[";
    while (current->next != head) {
        cout << current->val;
        cout << ", ";
        current = current->next;
    }
    cout << current->val << "]" << endl;

}

int length() {
    Node* current = head;
    int count = 0;
    
    while (current->next != head) {
        count++;
        current = current->next;
    }
    return count;
}





vector<int> josephus_sequence(int k){
vector<int> myvector;
Node *ptr1 = head, *ptr2 = head;


    while (ptr1->next != ptr1) 
    { 
      
        int count = 0; 
        while (count != k) 
        { 
            ptr2 = ptr1; 
            ptr1 = ptr1->next; 
            count++; 
        } 
  
        myvector.push_back (ptr1->val);
        ptr2->next = ptr1->next;
        ptr1 = ptr2->next; 
        
    } 

    myvector.push_back (ptr1->val);
    
    
     
 
            return myvector; 

}

};

void last_man_standing(int n, int k){
  
   CircularLinkedList list(n);
   vector<int> l;
   
   l=list.josephus_sequence(k);
   int p = l.size(); 

   
   cout << l[p - 1] << endl; 
   

  
 }






int main(){

//vector<int> v = {2,1};
//CircularLinkedList list(v);
// list.append(2);
// list.append(1);
//cout<<list[2]<<endl;

//list.print();
 
 int n=68;
 int k=6; 
 last_man_standing(n,k);


}