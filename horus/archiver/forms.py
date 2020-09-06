from django import forms  

class getform(forms.Form):  
    
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))
    CHOICES = [('eng', 'English'),('fra', 'French'), ('spa', 'Spanish'), ('ara', 'Arabic')]
    Language = forms.ChoiceField(widget=forms.RadioSelect, choices=CHOICES)

class handform(forms.Form):  
    
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))
"""    CHOICES = [('eng', 'English'),('fra', 'French'), ('spa', 'Spanish'), ('ara', 'Arabic')]
    Language = forms.ChoiceField(widget=forms.RadioSelect, choices=CHOICES)"""

class googleform(forms.Form):  
    
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))
    CHOICES = [("1", 'Faces and Gestures'),("2", 'Label Identification'), ("3", 'Landmark Detection'), ("4", 'Multi Label Detection')]
    choice = forms.ChoiceField(widget=forms.RadioSelect, choices=CHOICES)


class chequeform(forms.Form):  
    
    cheque = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))
    

class invoiceform(forms.Form):  
    
    invoice = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))
    

class kycform(forms.Form):  
    
    Doc = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))
   


   

