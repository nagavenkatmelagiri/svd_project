from django import forms

class UploadForm(forms.Form):
    image = forms.ImageField()
    k = forms.IntegerField(min_value=1, max_value=200, initial=50, label='k (singular values)')
