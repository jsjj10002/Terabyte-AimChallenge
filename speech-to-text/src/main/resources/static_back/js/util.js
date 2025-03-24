$(document).ready(function(){
	
})
/* =====================================================================================================================================
해당 util.js의 함수들은 예시임으로 
jsp에 넣어 쓰지 말고
다른 js 파일에 다른 함수 이름으로 사용할 것
===================================================================================================================================== */
// 업로드 파일 예시 (단일 파일 업로드 기준)
var upload_file = []; // 업로드에 필요한 파일을 담는 리스트
function fileUpload(e) {
	const image_file = $('#file_input').val();
	const image_file_name = image_file.split('\\')[image_file.split('\\').length-1];
	const files = e.target.files;
	const filesArr = Array.prototype.slice.call(files);
    if (filesArr.length > 0) {
		// upload_file = []; // 업로드가 하나 일 경우 파일이 바뀔때 마다 초기화
		filesArr.forEach(function(f){
			upload_file.push(f);
			// $('#file_name_input').val(image_file_name); // 파일 이름 표기 영역이 div (html) 나 input (val) 등으로 별도로 존재하는 경우
			var reader = new FileReader();
			/* reader.onload = function(e) { // 업로드 후 썸네일(미리보기)을 사용하고 싶을 때
				$('#thumnail').prop('src', e.target.result);
			}; */
			reader.readAsDataURL(f);	
		});
	}
	$('#file_input').val(null); // 파일 중복 선택시 자연스럽게 선택되게 하기 위함
}

/* 
FormData 전송시 전송할 formdata 에 아래와 같이 append 추가

	if (upload_file.length > 0) {
		for (var i = 0; i < upload_file.length; i++) {			
			formdata.append("file"+i, upload_file[i]);
		}
	}
	
*/
// ====================================================================================================
// sweetalert 사용시
// <script src="https://unpkg.com/sweetalert/dist/sweetalert.min.js"></script> <!-- jsp 추가 -->
// 그 외 기타 사용법은 https://sweetalert.js.org/guides/ 에서

// 단순 확인 버튼만 있는 alert창
function swal_alert() {
	// 변수는 title, text, icon 순 
	swal('ㅇㅇㅇ xx', 'ㅇㅇㅇ을 성공적으로 xx 했습니다.', 'success').then(function(){
		
	});
	// 콜백 할게 없다면 then은 생략 가능
	// swal('ㅇㅇㅇ xx', 'ㅇㅇㅇ을 성공적으로 xx 했습니다.', 'success');
}

// 버튼이 두개 이상인 alert창
function swal_confirm() {
	swal({
		title: 'ㅇㅇㅇ xx', // 상단 제목 영역 - 안쓴다면 null 값 말고 ''
		text: 'ㅇㅇㅇ을 xx 하시겠습니까?', // 내용 영역
		icon: 'warning', // 종류 (success - 초로색 체크, warning - 노란색 느낌표, error - 빨간색 X표시 등)
		buttons: { // '변수 : 버튼 이름' 형태에 위에서 부터 순서대로 좌측에서 우측으로 버튼이 생성
			cancel : '취소', 
			ok :  '확인'
		}
	}).then((value) => { // value 는 buttons에서 선언한 변수 명
		if (value == 'ok') { // ok 변수('확인') 버튼을 눌렀을때 실행
        	
		}
	})
}
// ====================================================================================================
 
