#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
using namespace std;




struct Node{
int val;
Node*next;
Node*prev;

Node(int n) {
    val = n;
    next = nullptr;
    prev = nullptr;
}

Node(int n, Node* p) {
    val = n;
    next = p;
}

};






class LinkedList{
private:

Node*head;
Node*tail;
public:

LinkedList(){


head=nullptr;
tail=nullptr;


  

}

LinkedList(vector<int>initial){

head=nullptr;
tail=nullptr;

  for(int e: initial){
        append(e);
    }

}


~LinkedList(){
  Node* current;
    Node* next;
    
    current = head;
    
    while (current != nullptr) {
        next = current->next;
        delete current;
        current = next;
    }

}

int length() {
    Node* current = head;
    int count = 0;
    
    while (current != nullptr) {
        count++;
        current = current->next;
    }
    return count;
}



void append(int val) {
    if (head == nullptr) {
        head= new Node(val);
        tail=head;
        return;
        }

    tail->next= new Node(val);
    tail=tail->next;
    

    }   

void print() {
    Node* current = head;
    cout << "[";
    while (current->next != nullptr) {
        cout << current->val;
        cout << ", ";
        current = current->next;
    }
    cout << current->val << "]" << endl;
}



Node* get_node(int index) {
    if (index < 0 or index >= length()) {
        throw range_error("IndexError: Index out of range");
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

 void insert(int val, int index) {
     if(index==0){
     Node* prev = get_node(index);
     Node* next = prev->next;
     prev->next = new Node(val, next); 

     }

     Node* prev = get_node(index-1);
     Node* next = prev->next;
     prev->next = new Node(val, next);
 }

 void remove(int index) {

    if(index==0){
    Node* prev = head;
    
     Node* next = prev->next;

     head=next;
  
    } 
    else{
    Node* prev = get_node(index-1);
    
     Node* next = prev->next;

     prev->next=next->next;
    }

}


int pop(int index){
    Node*current = get_node(index);
    if(index==0){
    Node* prev = head;
    
     Node* next = prev->next;

     head=next;
  
    } 
   else{
    Node* prev = get_node(index-1);
    Node* next = prev->next;

     prev->next=next->next;
   }
   return current->val;
}



int pop(){

    return tail->val;
}






};



int main(){

vector<int> v = {7,6,5,4,3,1};
LinkedList list(v);
list.remove(0);
list.pop(4);
cout<<list.pop()<<endl;
cout<<list[3]<<endl;
list.insert(0,2);
list.print();

}







