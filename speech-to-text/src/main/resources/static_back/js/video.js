var player;

// 기본 videojs 필수 css href, js src
// <link href="https://vjs.zencdn.net/7.19.2/video-js.css" rel="stylesheet" />
// <script src="https://vjs.zencdn.net/7.19.2/video.min.js"></script>
$(document).ready(function(){
	
	
	var videoId = "my-video"; // videojs가 video 태그의 id값을 토대로 생성함
	makeTrack($('#'+videoId),"/srt/test.vtt", "ko", "한국어", false);
	
	var vr = false;
	
	var controlBar = makeControlBar(true, true, false, true, true);
	var plugins = vr ? make3dPlugins() : null;
	var options = makeVideoOptions("/video/test.mp4", null, true, false, 960, 640, [0.5, 1, 1.5, 2], controlBar, plugins)
	
	var plugins2 = !vr ? make3dPlugins() : null;
	var options2 = makeVideoOptions("https://raw.githubusercontent.com/Pierrinho/elephant/master/elephant.mp4", null, true, false, 960, 640, [0.5, 1, 1.5, 2], controlBar, plugins2)
	
	player = videojs(videoId, options); // videojs 생성
	player = videojs(videoId+"2", options2); // videojs 생성
	
})

// 자막 추가 (videojs 생성 전에 할 것)
// 영상 태그 ($('')), 자막파일 경로 (String), 언어 (String - ex. "ko", "en"), 표기(String), 디폴트값 (boolean)
// defaultValue (true : 시작시 자막 켜두기, false : 시작시 꺼두기)
// 자막파일은 현재 vtt만 적용됨
function makeTrack(tag, src, srclang, label, defaultValue) {
	var track = '<track kind="commentary" src="'+src+'" srclang="'+srclang+'" label="'+label+'"';
	if (defaultValue) track = track + ' default="'+defaultValue+'"';
	track = track + '/>';
	tag.append(track);
}

// 기본 컨트롤바 설정
// 일시정지/재생 버튼 유무(boolean), pip모드 버튼 유무(boolean), 남은 시간 표시 여부[음수로 표기됨](boolean), 
// 재생 프로그레스바 유무(boolean), 전체화면 버튼 유무(boolean)
function makeControlBar(playToggle, pictureInPictureToggle, remainingTimeDisplay, progressControl, fullscreenToggle) {
	var controlbar = {
    	playToggle : playToggle, // 일시정지/재생버튼
    	pictureInPictureToggle : pictureInPictureToggle, // pip모드
    	remainingTimeDisplay : remainingTimeDisplay, // 남은시간 표시
    	progressControl : progressControl, // 프로그레스 바
    	fullscreenToggle : fullscreenToggle
    }
    return controlbar;
}

// 기본 3d 동영상 플러그인 - 아래의 js 필수 (없으면 작동 안함)
// <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r76/three.js"></script>
// <script src="https://rawgit.com/yanwsh/videojs-panorama/master/dist/videojs-panorama.v5.js"></script>
// 별도의 플러그인이 필요하다면 아래와 같이 객체({})로 생성하면 됨.
// 예시 : https://codepen.io/pierrinho/pen/dVZpWB
function make3dPlugins() {
	var plugins = {
		panorama: {
			clickAndDrag: true,
			clickToToggle: true,
			autoMobileOrientation: true,
			backToVerticalCenter: false,
			backToHorizonCenter: false
		}
	}
	return plugins;
}

// 기본 옵션 설정 
// 영상경로(String), 썸네일경로(String), 컨트롤바 유무(boolean), 음소거(boolean), 폭(Number), 높이(Number),
// 배속(List<Number>), 컨트롤바({}), 플러그인({})
function makeVideoOptions(src, poster, controls, mute, width, height, playbackRates, controlBar, plugins) {
	var splitSrc = src.split('.');
	var extension = splitSrc[splitSrc.length-1];
	var options = {
            sources : [
                { src : src, type : "video/"+extension}
            ],
            poster : poster, // 썸네일
            controls : controls, // 컨트롤바
            playsinline : true,
            language : "ko",
            muted : mute, // 음소거
            preload : "auto", // 영상 데이터 로드
            width : width, // 가로
            height : height, // 세로,
            playbackRates: playbackRates, // 배속
            controlBar : controlBar,
            plugins: plugins
        }
        
	return options;
}

// 기타 videojs 사이트
// videojs 가이드 리스트 : https://videojs.com/guides
// videojs 기본 가이드 : https://videojs.com/guides/options/#aspectratio
// video 스킨 : https://github.com/videojs/video.js/wiki/Skins
// video 스킨 커스텀 : https://www.scriptsmashup.com/Video_Skin_Generator/Videojs/videojs-skin-generator.html