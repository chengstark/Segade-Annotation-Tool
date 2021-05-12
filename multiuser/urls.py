from . import views
from django.urls import path

urlpatterns = [
    path('', views.login, name='multiuser_home'),
    path('process_credential', views.process_credential, name='multiuser_process_credential'),
    path('clear_anno', views.clear_anno, name='multiuser_clear'),
    path('submit_anno', views.submit_anno, name='multiuser_submit'),
    path('export_anno', views.export_anno, name='multiuser_export'),
    path('no_artifact', views.no_artifact, name='multiuser_no_artifact'),
    path('all_artifact', views.all_artifact, name='multiuser_all_artifact'),
    path('skip', views.skip, name='multiuser_skip'),
    path('redo_last', views.redo_last, name='multiuser_redo_last'),
    path('last_page_user', views.last_page_user, name='multiuser_your_last'),
    path('next_page_user', views.next_page_user, name='multiuser_your_next'),
    path('viz_export_anno_user', views.viz_export_anno_user, name='multiuser_your_viz_export'),
    path('inspect_index_user', views.inspect_index_user, name='multiuser_your_viz_inspect_idx'),
    path('reanno_index_user', views.reanno_index_user, name='multiuser_your_viz_reanno'),
    path('reanno_clear_anno_user', views.reanno_clear_anno_user, name='multiuser_your_clear_reanno'),
    path('reanno_export_anno_user', views.reanno_export_anno_user, name='multiuser_your_export_reanno'),
    path('reanno_no_artifact_user', views.reanno_no_artifact_user, name='multiuser_no_artifact_viz_reanno'),
    path('reanno_all_artifact_user', views.reanno_all_artifact_user, name='multiuser_your_all_artifact_reanno'),
    path('reanno_submit_anno_user', views.reanno_submit_anno_user, name='multiuser_your_submit_reanno'),
    path('back_to_viz', views.back_to_viz_handler, name='multiuser_your_viz_handler'),
    path('log_out', views.logout, name='multiuser_log_out'),
    path('viz_user', views.viz_user, name='multiuser_inspect_your_anno_out'),
    path('back_to_annotate_user', views.back_to_annotate_user, name='multiuser_back_to_your_anno_out'),
    path('anno_user', views.anno_home, name='multiuser_anno_user'),
    path('viz_all', views.viz_all, name='multiuser_viz_all'),
    path('last_all_user', views.last_page_all_user, name='multiuser_last_viz_all'),
    path('next_all_user', views.next_page_all_user, name='multiuser_next_viz_all'),
    path('viz_export_anno_all_user', views.viz_export_anno_all_user, name='multiuser_export_viz_all'),
    path('inspect_index_all_user', views.inspect_index_all_user, name='multiuser_inspect_index_all_user'),
    path('reanno_index_all_user', views.reanno_index_all_user, name='multiuser_reanno_index_all_user'),
    path('back_to_menu', views.back_to_menu, name='multiuser_back_to_menu'),
    path('sort_anno_all_user', views.anno_sorting_handler_all_user, name='multiuser_sort_anno_all_user'),
]


