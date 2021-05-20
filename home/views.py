from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.db import connection
from django import forms
import os
import json
import requests

def MainPage(request):

    if request.method == 'GET':
        cursor = connection.cursor()
        cnt = cursor.execute("SELECT * FROM store")
        store_query = cursor.fetchall()
        store_query = list(store_query)
        store_query.reverse()
        store_list_dict = {}
        store_list = []
        while store_query:
            store = store_query.pop()
            if store[3] == 0:
                t = '보관중'
            else:
                t = '보관 가능'
            store_list.append({'id': store[0], 'state': t})
        return render(request,'shop-grid.html',{'store_list': store_list})
def StorePage(request):

    if request.method == 'GET':
        # cursor = connection.cursor()
        # cnt = cursor.execute("UPDATE store SET 's_dung'={}, 's_ho'={},'pw'={},'state'={} WHERE 'id'={}"
        #                      .format())
        # store_query = cursor.fetchall()
        i = 0
        dung = 0
        print(request.GET.getlist('dung'))
        for item in request.GET.getlist('dung'):
            print(item)
            if item != '':
                dung = int(item)
            else:
                i+=1
        id = i
        print(dung)
        print(id)
        res = requests.get('http://localhost:8000/controller/user/1/?'
                           'front_sub_motor='+str(dung) +'&'
                           'back_sub_motor=100')
        print(res.text)
        return redirect('main_page')
