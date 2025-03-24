$(document).ready(function(){
	/*const data = {page : 1, size : 10};
	const aoo = ajax_object_options("POST", "/auth/callbackSample", data);
	ajax(aoo, callback_function, callback_function)*/
})

// 객체형 options (ApiController에서 @RequestBody로 받음)
function ajax_object_options(type, url, data) {
	var options = {
		type: type,
		url : url,
		data : JSON.stringify(data),
		contentType: "application/json; charset=utf-8",
		dataType: "json"
	}
	
	return options;
}

// DataForm 형태 options (ApiController에서 @ModelAttribute로 받음)
function ajax_formdata_options(type, url, formdata) {
	var options = {
		type: type,
		enctype: 'multipart/form-data',
		url: url,
		data: formdata,
		processData: false,
		contentType: false,
		cache: false,
		timeout: 600000,
	}
	
	return options;
}

// ajax
function ajax(options, doneCallback, failCallback) {
	$.ajax(options).done(function(resp) {
		doneCallback(resp)
	}).fail(function(error){
		failCallback(error)
	})
}

// 콜백 함수 예시 (자유롭게 커스텀 하면 됨 / resp는 java에서 돌려받은 값[responseDto / returnDto])
function callback_function(resp) {
	console.log(resp)
}

// 필터링
function decodeHTMLEntities(str) {
	if (str !== undefined && str !== null && str !== '') {
		str = String(str);
		str = str.replace(/<script[^>]*>([\S\s]*?)<\/script>/gmi, '');
		str = str.replace(/<script[^>]*>([\S\s]*?)<\/script>/gmi, '');
	}
}