from django import forms

class UploadForm(forms.Form):
    image = forms.ImageField()
    k = forms.IntegerField(min_value=1, max_value=200, initial=50, label='k (singular values)')
    SIZE_CHOICES = [
        ('1.0', 'Large (100%)'),
        ('0.75', 'Medium (75%)'),
        ('0.5', 'Small (50%)'),
        ('0.25', 'Tiny (25%)'),
    ]
    size = forms.ChoiceField(
        choices=SIZE_CHOICES,
        initial='1.0',
        label='Output size',
        widget=forms.Select(attrs={'id': 'sizeSelect'})
    )
    sharpen = forms.BooleanField(required=False, initial=False, label='Apply sharpen')
