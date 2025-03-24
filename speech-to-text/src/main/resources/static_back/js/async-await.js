$(document).ready(function(){
    var data = {page : 1, size : 10};
	var aoo = ajax_object_options("POST", "/auth/callbackSample", data);
		
	logicFunction(aoo);
})

function asyncAjax(aoo) {
  return new Promise((resolve, reject) => {
	$.ajax(aoo).done(function(resp) {
		resolve(resp);
	}).fail(function(error){
		reject(error);
	})
  });
}

function logicFunction(aoo) {
	asyncAjax(aoo).then((resp) => {		
		console.log(resp);
	}).catch((error) => {
		console.log(error);
	});
}