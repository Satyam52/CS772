import json
import requests
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from .auth import get_auth_token, is_valid_token
from .models import Cycle, AppUser
from .serializer import *
from .auth import handle_user
from .pos_tag_bak1.inferences import Inference
from .cbow.get_neighbours import CBOWInference
from .sst.inference import Inference as SSTInference


class RNNApiView(APIView):
    model = Inference()
    
    def get(self, request):        
        return Response({"msg": "success"}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        text = request.data['input']
        output = self.model.get_tags(text)
        token_inp = self.model.preprocess_sentence(text)
        print(token_inp, output)
        # res = self.model.inference(text=text)
        
        return Response({'input':token_inp, "output":output}, status=status.HTTP_200_OK)
    
class CBOWApiView(APIView):
    model = CBOWInference()
    
    def get(self, request):        
        return Response({"msg": "success"}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        text = request.data['words']
        output = self.model.infer(text)        
        return Response({'words':output}, status=status.HTTP_200_OK)
    

class ProjectApiView(APIView):
    model = SSTInference()
    
    def get(self, request):        
        return Response({"msg": "success"}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        text = request.data['input']
        output = self.model.infer(text)
        print(output)
        # token_inp = self.model.preprocess_sentence(text)
        # print(token_inp, output)
        # res = self.model.inference(text=text)
        
        return Response({"scores":output}, status=status.HTTP_200_OK)