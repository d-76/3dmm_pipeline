#ifndef brec_processes_h_
#define brec_processes_h_

#include <bprb/bprb_macros.h>
#include <bprb/bprb_func_process.h>

// execute and const functions
DECLARE_FUNC_CONS(brec_bayesian_update_process);
DECLARE_FUNC_CONS(brec_change_area_process);
DECLARE_FUNC_CONS(brec_create_mog_image_process);
DECLARE_FUNC_CONS(brec_density_to_prob_map_process);
DECLARE_FUNC_CONS(brec_glitch_overlay_process);
DECLARE_FUNC_CONS(brec_glitch_process);
DECLARE_FUNC_CONS(brec_prob_map_area_process);
DECLARE_FUNC_CONS(brec_prob_map_roc_compute_process);
DECLARE_FUNC_CONS(brec_prob_map_supress_process);
DECLARE_FUNC_CONS(brec_prob_map_threshold_process);
DECLARE_FUNC_CONS(brec_update_changes_process);
DECLARE_FUNC_CONS(brec_recognize_structure_process);

DECLARE_FUNC_CONS(brec_recognize_structure2_process);
DECLARE_FUNC_CONS(brec_construct_bg_op_models_process);
DECLARE_FUNC_CONS_INIT(brec_construct_class_op_models_process);
DECLARE_FUNC_CONS(brec_create_hierarchy_process);
DECLARE_FUNC_CONS(brec_load_hierarchy_process);
DECLARE_FUNC_CONS(brec_save_hierarchy_process);

DECLARE_FUNC_CONS(brec_learner_layer0_init_process);
DECLARE_FUNC_CONS(brec_learner_layer0_fit_process);
DECLARE_FUNC_CONS(brec_learner_layer0_rank_process);
DECLARE_FUNC_CONS_INIT(brec_learner_layer0_update_posterior_stats_process);
DECLARE_FUNC_CONS_INIT(brec_learner_layer0_update_stats_process);

DECLARE_FUNC_CONS(brec_learner_layer_n_init_process);
DECLARE_FUNC_CONS(brec_learner_layer_n_update_stats_process);
DECLARE_FUNC_CONS(brec_learner_layer_n_fit_process);

DECLARE_FUNC_CONS(brec_draw_hierarchy_process);

DECLARE_FUNC_CONS(brec_initialize_detector_process);
DECLARE_FUNC_CONS(brec_add_hierarchy_to_detector_process);
DECLARE_FUNC_CONS_INIT(brec_detect_hierarchy_process);

DECLARE_FUNC_CONS(brec_set_hierarchy_model_dir_process);

#endif  // brec_processes_h_
