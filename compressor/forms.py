from django import forms

class UploadForm(forms.Form):
    # Compression method selection
    METHOD_CHOICES = [
        ('svd', 'SVD (Classical Linear Algebra)'),
        ('cnn', 'CNN Autoencoder (Deep Learning)'),
    ]
    method = forms.ChoiceField(
        choices=METHOD_CHOICES,
        initial='svd',
        label='Compression Method',
        widget=forms.Select(attrs={'id': 'methodSelect', 'onchange': 'toggleMethodFields()'})
    )
    
    image = forms.ImageField()
    
    # SVD parameters
    k = forms.IntegerField(
        min_value=1, 
        max_value=200, 
        initial=50, 
        label='k (singular values)',
        widget=forms.NumberInput(attrs={'class': 'svd-field'})
    )
    maintain_quality = forms.BooleanField(
        required=False, 
        initial=False, 
        label='Maintain clarity (auto k)',
        widget=forms.CheckboxInput(attrs={'class': 'svd-field'})
    )
    target_psnr = forms.IntegerField(
        min_value=10, 
        max_value=60, 
        initial=30, 
        label='Target PSNR (dB)', 
        required=False,
        widget=forms.NumberInput(attrs={'class': 'svd-field'})
    )
    
    # CNN parameters
    latent_dim = forms.IntegerField(
        min_value=16, 
        max_value=256, 
        initial=64, 
        label='Latent Dimension (lower = more compression)',
        widget=forms.NumberInput(attrs={'class': 'cnn-field'}),
        required=False
    )
    MODEL_TYPE_CHOICES = [
        ('standard', 'Standard (Better Quality)'),
        ('lightweight', 'Lightweight (Faster)'),
    ]
    model_type = forms.ChoiceField(
        choices=MODEL_TYPE_CHOICES,
        initial='standard',
        label='Model Type',
        widget=forms.Select(attrs={'class': 'cnn-field'}),
        required=False
    )
    
    # Common parameters
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
