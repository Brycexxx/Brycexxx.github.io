---
title: Bryce 的留言板
date: 2018-08-30 17:07:19
---
<img src='/assets/background.jpg' alt='background' height=300px width='100%' />
<div id="vcomment" class="comment"></div>
<script src="//cdn.jsdelivr.net/npm/jquery@latest/dist/jquery.min.js"></script>
<script src="//cdn.jsdelivr.net/npm/leancloud-storage@latest/dist/av-min.js"></script>
<script src='//cdn.jsdelivr.net/npm/valine@latest/dist/Valine.min.js'></script>
<script>
   var notify = '<%= theme.valine.notify %>' == true ? true : false;
   var verify = '<%= theme.valine.verify %>' == true ? true : false;
   new Valine({
            av: AV,
            el: '#vcomment',
            notify: notify,
            verify: verify,
            app_id: "<%= theme.valine.appid %>",
            app_key: "<%= theme.valine.appkey %>",
            placeholder: "由于leancloud体验实例有每天 6 小时的强制休眠时间，所以大家的留言未必能及时知晓，望谅解。若事情紧急，还请邮件联系，谢谢",
            avatar: "<%= theme.valine.avatar %>",
            avatar_cdn: "<%= theme.valine.avatar_cdn %>",
            pageSize: <%= theme.valine.pageSize %>
    });
</script>