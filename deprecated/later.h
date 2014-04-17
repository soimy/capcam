//
//  later.h
//  capcam
//
//  Created by 沈 一鸣 on 14-4-7.
//  Copyright (c) 2014年 SYM. All rights reserved.
//

#ifndef __capcam__later__
#define __capcam__later__

#include <iostream>
#include <iostream>
#include <string.h>
#include <chrono>
#include <future>

using namespace std;

class later{
    public:
        template <class callable, class... arguments>  later(int after, bool async, int loop, callable&& f, arguments&&... args){
            function<typename std::result_of<callable(arguments...)>::type()> task(bind(forward<callable>(f), forward<arguments>(args)...));
            if (async){
                thread([after, task, loop]() {
                    if (loop == 0) {
                        while (true) {
                            this_thread::sleep_for(std::chrono::milliseconds(after));
                            task();
                        }
                    }else{
                        for (int i = loop; i >= 0; --i) {
                            this_thread::sleep_for(std::chrono::milliseconds(after));
                            task();
                        }
                    }
                }).detach();
            }
            else{
                if(loop > 0)
                    for (int i = loop; i >=0; --i){
                        this_thread::sleep_for(std::chrono::milliseconds(after));
                        task();
                    }
            }
        }
};

#endif /* defined(__capcam__later__) */
