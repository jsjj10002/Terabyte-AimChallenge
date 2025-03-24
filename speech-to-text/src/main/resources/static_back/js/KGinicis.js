function on_pay() {
	myform = document.mobileweb;
	myform.action = "https://mobile.inicis.com/smart/payment/";
	myform.target = "_self";
	myform.submit();
}