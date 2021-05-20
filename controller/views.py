from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
node_mcu_map = [
    {'mcu_id':1, 'front_sub_motor': 30, 'back_sub_motor': 1},
    {'mcu_id':2, 'front_sub_motor': 1, 'back_sub_motor': 1},
    {'mcu_id':3, 'front_sub_motor': 1, 'back_sub_motor': 1},
    {'mcu_id':4, 'front_sub_motor': 1, 'back_sub_motor': 1}
]

@csrf_exempt
def NodeMCUController(request, id):
    global node_mcu_map
    if request.method == 'GET':
        front_sub_motor = 0
        back_sub_motor = 0
        for node in node_mcu_map:
            if node['mcu_id'] == id:
                front_sub_motor = node['front_sub_motor']
                back_sub_motor = node['back_sub_motor']

        return JsonResponse({'front_sub_motor': front_sub_motor,
                             'back_sub_motor': back_sub_motor
                             })

@csrf_exempt
def DeviceContorller(request, id):
    global node_mcu_map
    if request.method == 'GET':
        for node in node_mcu_map:
            if node['mcu_id'] == id:
                node['front_sub_motor'] = int(request.GET.get('front_sub_motor'))
                node['back_sub_motor'] = int(request.GET.get('back_sub_motor'))
        return JsonResponse({'front_sub_motor': int(request.GET.get('front_sub_motor')),
                             'back_sub_motor': int(request.GET.get('back_sub_motor'))
                             })


