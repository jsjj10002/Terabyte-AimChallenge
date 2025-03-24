$(function() {
    // 마이크 경로 변경
    $(document).on('click','.mic',function() {
        const micSrc = $(this).children('img')
        if (!$(this).hasClass('on')) {
            micSrc.attr('src','/img/on.png');
            $(this).addClass('on');

            /*setTimeout(() => {
                $('.contents').css('display', 'none');
                $('.payment').css('display', 'flex');
    
                setTimeout(() => {
                    $('.payment').css('display', 'none');
                    $('.paymentEnd').css('display', 'flex');
                }, 2000);
            }, 1000);*/
        } else {
            micSrc.attr('src','/img/off.png');
            $(this).removeClass('on');
        }
    });


});