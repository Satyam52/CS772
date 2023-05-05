from django.urls import path

from .views import *

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path("rnn", RNNApiView.as_view(), name='RNN'),
    path("activation", ProjectApiView.as_view(), name='Activation'),
    path("cbow", CBOWApiView.as_view(), name='CBOW'),
]
