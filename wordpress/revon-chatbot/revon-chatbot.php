<?php
/**
 * Plugin Name: RevOn Policy Chatbot
 * Description: Secure public policy PDF management and backend integration for RevOn Business Solutions.
 * Version: 1.5.2
 * Author: RevOn Business Solutions
 * Text Domain: revon-policy-chatbot
 */

if (!defined('ABSPATH')) {
    exit;
}

define('REVON_PCB_VERSION', '1.5.2');

register_activation_hook(__FILE__, 'revon_pcb_activate');


/* =========================================================
 * 1. Database and private storage
 * ======================================================= */

function revon_pcb_table() {
    global $wpdb;
    return $wpdb->prefix . 'revon_documents';
}
function revon_pcb_operations_table() {
    global $wpdb;
    return $wpdb->prefix . 'revon_operations';
}

function revon_pcb_private_dir() {
    return trailingslashit(dirname(untrailingslashit(ABSPATH))) . 'revon-private-documents';
}

function revon_pcb_activate() {
    global $wpdb;

    require_once ABSPATH . 'wp-admin/includes/upgrade.php';

    $table = revon_pcb_table();
    $charset = $wpdb->get_charset_collate();

    $sql = "CREATE TABLE {$table} (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        document_id VARCHAR(100) NOT NULL,
        logical_document_id VARCHAR(100) NOT NULL,
        filename VARCHAR(255) NOT NULL,
        file_hash CHAR(64) NOT NULL,
        version INT UNSIGNED NOT NULL DEFAULT 1,
        storage_path TEXT NOT NULL,
        total_pages INT UNSIGNED NOT NULL DEFAULT 0,
        total_chunks INT UNSIGNED NOT NULL DEFAULT 0,
        status VARCHAR(30) NOT NULL DEFAULT 'processing',
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        PRIMARY KEY  (id),
        UNIQUE KEY document_id (document_id),
        UNIQUE KEY unique_policy_version (logical_document_id, version),
        KEY file_hash (file_hash),
        KEY status (status),
        KEY logical_document_id (logical_document_id)
    ) {$charset};";

    dbDelta($sql);

    $ops = revon_pcb_operations_table();
    $ops_sql = "CREATE TABLE {$ops} (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        request_id CHAR(36) NOT NULL,
        operation_type VARCHAR(20) NOT NULL,
        document_id VARCHAR(100) NULL,
        previous_document_id VARCHAR(100) NULL,
        logical_document_id VARCHAR(100) NULL,
        version INT UNSIGNED NULL,
        status VARCHAR(40) NOT NULL DEFAULT 'pending',
        payload_hash CHAR(64) NULL,
        expected_vectors INT UNSIGNED NULL,
        attempt_count INT UNSIGNED NOT NULL DEFAULT 0,
        last_error TEXT NULL,
        result_json LONGTEXT NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        PRIMARY KEY  (id),
        UNIQUE KEY request_id (request_id),
        KEY document_id (document_id),
        KEY logical_document_id (logical_document_id),
        KEY operation_status (operation_type, status)
    ) {$charset};";
    dbDelta($ops_sql);

    update_option('revon_pcb_db_version', REVON_PCB_VERSION);

    $dir = revon_pcb_private_dir();

    if (!file_exists($dir)) {
        wp_mkdir_p($dir);
    }

    if (is_dir($dir)) {
        if (!file_exists($dir . '/.htaccess')) {
            @file_put_contents($dir . '/.htaccess', "Require all denied\nDeny from all\n");
        }

        if (!file_exists($dir . '/index.php')) {
            @file_put_contents($dir . '/index.php', "<?php\nhttp_response_code(404); exit;\n");
        }

        if (!file_exists($dir . '/web.config')) {
            @file_put_contents(
                $dir . '/web.config',
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><configuration><system.webServer><authorization><deny users=\"*\" /></authorization></system.webServer></configuration>"
            );
        }
    }
}


/* =========================================================
 * 2. Admin menu and settings
 * ======================================================= */

add_action('admin_menu', function () {
    add_menu_page(
        'RevOn Policy Chatbot',
        'RevOn Chatbot',
        'manage_options',
        'revon-policy-chatbot',
        'revon_pcb_dashboard',
        'dashicons-format-chat',
        25
    );

    add_submenu_page(
        'revon-policy-chatbot',
        'Documents',
        'Documents',
        'manage_options',
        'revon-pcb-documents',
        'revon_pcb_documents'
    );

    add_submenu_page(
        'revon-policy-chatbot',
        'Settings',
        'Settings',
        'manage_options',
        'revon-pcb-settings',
        'revon_pcb_settings'
    );
});

add_action('admin_init', function () {
    register_setting('revon_pcb_settings', 'revon_pcb_api_base_url', array(
        'sanitize_callback' => 'esc_url_raw'
    ));

    register_setting('revon_pcb_settings', 'revon_pcb_admin_token', array(
        'sanitize_callback' => 'sanitize_text_field'
    ));

    // Keep the public-chat credential separate from the document-admin token.
    // An empty settings field preserves the previously saved secret.
    register_setting('revon_pcb_settings', 'revon_pcb_chat_token', array(
        'sanitize_callback' => 'revon_pcb_preserve_secret_when_blank'
    ));
});

function revon_pcb_preserve_secret_when_blank($value) {
    $value = is_string($value) ? trim($value) : '';
    return $value !== '' ? $value : (string) get_option('revon_pcb_chat_token', '');
}


/* =========================================================
 * 3. Admin notices and dashboard
 * ======================================================= */

function revon_pcb_notice() {
    if (empty($_GET['revon_msg'])) {
        return;
    }

    $type = (
        isset($_GET['revon_type']) &&
        sanitize_key(wp_unslash($_GET['revon_type'])) === 'error'
    ) ? 'error' : 'success';

    echo '<div class="notice notice-' . esc_attr($type) . ' is-dismissible"><p>' .
        esc_html(wp_unslash($_GET['revon_msg'])) .
        '</p></div>';
}

function revon_pcb_redirect($message, $type = 'success') {
    $redirect = admin_url('admin.php?page=revon-pcb-documents');

    wp_safe_redirect(add_query_arg(array(
        'revon_type' => $type,
        'revon_msg' => $message
    ), $redirect));
    exit;
}

function revon_pcb_dashboard() {
    if (!current_user_can('manage_options')) {
        return;
    }

    global $wpdb;

    $stats = $wpdb->get_row(
        "SELECT
            COUNT(*) AS total,
            COALESCE(SUM(status = 'active'), 0) AS active,
            COALESCE(SUM(status = 'processing'), 0) AS processing,
            COALESCE(SUM(status = 'failed'), 0) AS failed,
            COALESCE(SUM(total_chunks), 0) AS chunks
         FROM " . revon_pcb_table(),
        ARRAY_A
    );
    ?>
    <div class="wrap">
        <h1>RevOn Policy Chatbot</h1>
        <?php revon_pcb_notice(); ?>

        <p>Manage approved public RevOn policy documents.</p>

        <div style="display:flex;gap:16px;flex-wrap:wrap">
            <?php
            foreach (array(
                'Total versions' => 'total',
                'Active versions' => 'active',
                'Processing' => 'processing',
                'Failed' => 'failed',
                'Indexed chunks' => 'chunks'
            ) as $label => $key):
            ?>
                <div style="background:#fff;border:1px solid #ddd;padding:18px;min-width:130px">
                    <strong style="font-size:26px">
                        <?php echo esc_html((string) ($stats[$key] ?? 0)); ?>
                    </strong>
                    <br>
                    <?php echo esc_html($label); ?>
                </div>
            <?php endforeach; ?>
        </div>

        <p>
            <a class="button button-primary" href="<?php echo esc_url(admin_url('admin.php?page=revon-pcb-documents')); ?>">Manage documents</a>
            <a class="button" href="<?php echo esc_url(admin_url('admin.php?page=revon-pcb-settings')); ?>">Settings</a>
        </p>

        <div class="notice notice-info inline">
            <p>
                Document administration uses the separate backend admin token.
                Public visitors can use the chatbot without a WordPress account; the server-side chat token is configured separately in Settings.
            </p>
        </div>
    </div>
    <?php
}

function revon_pcb_settings() {
    if (!current_user_can('manage_options')) {
        return;
    }
    ?>
    <div class="wrap">
        <h1>RevOn Chatbot Settings</h1>
        <?php revon_pcb_notice(); ?>

        <form method="post" action="options.php">
            <?php settings_fields('revon_pcb_settings'); ?>

            <table class="form-table">
                <tr>
                    <th>
                        <label for="revon_pcb_api_base_url">FastAPI base URL</label>
                    </th>
                    <td>
                        <input
                            class="regular-text"
                            type="url"
                            id="revon_pcb_api_base_url"
                            name="revon_pcb_api_base_url"
                            value="<?php echo esc_attr(get_option('revon_pcb_api_base_url', '')); ?>"
                            placeholder="https://your-backend.onrender.com"
                        >
                    </td>
                </tr>

                <tr>
                    <th>
                        <label for="revon_pcb_admin_token">Backend admin token</label>
                    </th>
                    <td>
                        <input
                            class="regular-text"
                            type="password"
                            autocomplete="new-password"
                            id="revon_pcb_admin_token"
                            name="revon_pcb_admin_token"
                            value="<?php echo esc_attr(get_option('revon_pcb_admin_token', '')); ?>"
                        >
                        <p class="description">
                            Stored server-side. Never expose this token in browser JavaScript.
                        </p>
                    </td>
                </tr>

                <tr>
                    <th>
                        <label for="revon_pcb_chat_token">Backend public-chat token</label>
                    </th>
                    <td>
                        <input
                            class="regular-text"
                            type="password"
                            autocomplete="new-password"
                            id="revon_pcb_chat_token"
                            name="revon_pcb_chat_token"
                            value=""
                            placeholder="<?php echo get_option('revon_pcb_chat_token', '') ? esc_attr('Saved — leave blank to keep') : esc_attr('Paste the Render REVON_CHAT_TOKEN'); ?>"
                        >
                        <p class="description">
                            Separate from the admin token. Stored in WordPress and sent only from the server to FastAPI. Never place it in page HTML or JavaScript. Leave blank to keep the saved value.
                        </p>
                    </td>
                </tr>
            </table>

            <?php submit_button(); ?>
        </form>
    </div>
    <?php
}


/* =========================================================
 * 4. Document registry helpers and upload UI
 * ======================================================= */

function revon_pcb_get_active_policies() {
    global $wpdb;

    $table = revon_pcb_table();

    return $wpdb->get_results(
        "SELECT
            logical_document_id,
            MAX(version) AS latest_version,
            MAX(CASE WHEN status = 'active' THEN document_id ELSE NULL END) AS active_document_id,
            MAX(CASE WHEN status = 'active' THEN filename ELSE NULL END) AS active_filename
         FROM {$table}
         GROUP BY logical_document_id
         HAVING active_document_id IS NOT NULL
         ORDER BY MAX(updated_at) DESC",
        ARRAY_A
    );
}

function revon_pcb_documents() {
    if (!current_user_can('manage_options')) {
        return;
    }

    global $wpdb;

    $table = revon_pcb_table();
    $policies = revon_pcb_get_active_policies();

    $rows = $wpdb->get_results(
        "SELECT * FROM {$table} ORDER BY created_at DESC LIMIT 100",
        ARRAY_A
    );
    ?>
    <div class="wrap">
        <h1>Policy Documents</h1>
        <?php revon_pcb_notice(); ?>

        <h2>Upload Policy PDF</h2>
        <p>PDF only; maximum 10 MB. Exact duplicate files are rejected.</p>

        <form
            method="post"
            action="<?php echo esc_url(admin_url('admin-post.php')); ?>"
            enctype="multipart/form-data"
        >
            <input type="hidden" name="action" value="revon_pcb_upload">
            <?php wp_nonce_field('revon_pcb_upload'); ?>

            <table class="form-table">
                <tr>
                    <th><label for="revon_upload_type">Upload Type</label></th>
                    <td>
                        <select name="upload_type" id="revon_upload_type" required>
                            <option value="new_policy">New Policy</option>
                            <option value="new_version">New Version</option>
                        </select>
                        <p class="description">
                            Choose a new policy or an update to an existing policy.
                        </p>
                    </td>
                </tr>

                <tr id="revon_existing_policy_row">
                    <th><label for="revon_logical_document_id">Existing Policy</label></th>
                    <td>
                        <select name="logical_document_id" id="revon_logical_document_id">
                            <option value="">Select an existing active policy</option>
                            <?php foreach ($policies as $policy): ?>
                                <option value="<?php echo esc_attr($policy['logical_document_id']); ?>">
                                    <?php
                                    echo esc_html(
                                        $policy['active_filename'] .
                                        ' (Latest version: ' .
                                        $policy['latest_version'] . ')'
                                    );
                                    ?>
                                </option>
                            <?php endforeach; ?>
                        </select>
                        <p class="description">
                            Required only for New Version.
                        </p>
                    </td>
                </tr>

                <tr>
                    <th><label for="revon_policy_pdf">Policy PDF</label></th>
                    <td>
                        <input
                            type="file"
                            name="policy_pdf"
                            id="revon_policy_pdf"
                            accept=".pdf,application/pdf"
                            required
                        >
                    </td>
                </tr>
            </table>

            <?php submit_button('Upload and index PDF', 'primary', 'submit', true); ?>
        </form>

        <script>
        (function () {
            const typeField = document.getElementById('revon_upload_type');
            const policyRow = document.getElementById('revon_existing_policy_row');
            const policyField = document.getElementById('revon_logical_document_id');

            function updatePolicyField() {
                const isNewVersion = typeField.value === 'new_version';
                policyRow.style.display = isNewVersion ? '' : 'none';
                policyField.required = isNewVersion;
                if (!isNewVersion) {
                    policyField.value = '';
                }
            }

            typeField.addEventListener('change', updatePolicyField);
            updatePolicyField();
        })();
        </script>

        <hr>

        <h2>Document Registry</h2>
        <table class="widefat striped">
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Document ID</th>
                    <th>Logical Document ID</th>
                    <th>Version</th>
                    <th>Pages</th>
                    <th>Chunks</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <?php if (!$rows): ?>
                    <tr><td colspan="9">No documents uploaded.</td></tr>
                <?php else: ?>
                    <?php foreach ($rows as $row): ?>
                        <tr>
                            <td><?php echo esc_html($row['filename']); ?></td>
                            <td><code><?php echo esc_html($row['document_id']); ?></code></td>
                            <td><code><?php echo esc_html($row['logical_document_id']); ?></code></td>
                            <td><?php echo esc_html((string) $row['version']); ?></td>
                            <td><?php echo esc_html((string) $row['total_pages']); ?></td>
                            <td><?php echo esc_html((string) $row['total_chunks']); ?></td>
                            <td><?php echo esc_html($row['status']); ?></td>
                            <td><?php echo esc_html($row['created_at']); ?></td>
                            <td>
                                <?php if ($row['status'] === 'active'): ?>
                                    <form method="post"
                                          action="<?php echo esc_url(admin_url('admin-post.php')); ?>"
                                          onsubmit="return confirm('Deactivate this policy? It will immediately stop being eligible for chatbot retrieval. Its PDF and Pinecone vectors will be retained. Continue only if you intend to take this policy out of service.');">
                                        <input type="hidden" name="action" value="revon_pcb_deactivate_document">
                                        <input type="hidden" name="document_id" value="<?php echo esc_attr($row['document_id']); ?>">
                                        <?php wp_nonce_field('revon_pcb_deactivate_' . $row['document_id']); ?>
                                        <button type="submit" class="button button-small">Deactivate</button>
                                    </form>
                                <?php elseif ($row['status'] === 'inactive' || $row['status'] === 'failed'): ?>
                                    <form method="post"
                                          action="<?php echo esc_url(admin_url('admin-post.php')); ?>"
                                          onsubmit="return confirm('Permanently delete this inactive/failed document, its Pinecone vectors and private PDF? This cannot be undone.');">
                                        <input type="hidden" name="action" value="revon_pcb_delete_document">
                                        <input type="hidden" name="document_id" value="<?php echo esc_attr($row['document_id']); ?>">
                                        <?php wp_nonce_field('revon_pcb_delete_' . $row['document_id']); ?>
                                        <button type="submit" class="button button-small button-link-delete">Delete</button>
                                    </form>
                                <?php else: ?>
                                    <span class="description">Deletion unavailable — reconcile lifecycle state first</span>
                                <?php endif; ?>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                <?php endif; ?>
            </tbody>
        </table>
    </div>
    <?php
}


/* =========================================================
 * 5. Backend HTTP helper
 * ======================================================= */

function revon_pcb_backend_config() {
    $base = untrailingslashit((string) get_option('revon_pcb_api_base_url', ''));
    $token = (string) get_option('revon_pcb_admin_token', '');

    if (!$base || !wp_http_validate_url($base) || strpos($base, 'https://') !== 0) {
        return new WP_Error('invalid_backend_url', 'Configure a valid HTTPS FastAPI base URL in Settings.');
    }

    if (!$token) {
        return new WP_Error('missing_backend_token', 'Configure the backend admin token in Settings.');
    }

    if (!function_exists('curl_init')) {
        return new WP_Error('missing_curl', 'PHP cURL extension is required.');
    }

    return array(
        'base' => $base,
        'token' => $token
    );
}

/**
 * Convert FastAPI's detail field to safe, readable text.
 * FastAPI may return detail as either a string or a structured JSON object.
 */
function revon_pcb_backend_error_detail($data, $fallback) {
    if (!is_array($data) || !array_key_exists('detail', $data)) {
        return $fallback;
    }

    $detail = $data['detail'];

    if (is_string($detail) || is_numeric($detail) || is_bool($detail)) {
        return sanitize_text_field((string) $detail);
    }

    if (is_array($detail) || is_object($detail)) {
        $encoded = wp_json_encode($detail);
        return $encoded
            ? sanitize_text_field($encoded)
            : $fallback;
    }

    return $fallback;
}

function revon_pcb_backend_post_form($url, $token, $fields) {
    $ch = curl_init($url);

    if ($ch === false) {
        return new WP_Error(
            'backend_connection_error',
            'Could not initialize the backend connection.'
        );
    }

    $configured = curl_setopt_array($ch, array(
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $fields,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_CONNECTTIMEOUT => 15,
        CURLOPT_TIMEOUT => 180,
        CURLOPT_HTTPHEADER => array(
            'X-RevOn-Admin-Token: ' . $token,
            'Accept: application/json'
        ),
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_SSL_VERIFYHOST => 2
    ));

    if (!$configured) {
        curl_close($ch);
        return new WP_Error(
            'backend_connection_error',
            'Could not configure the backend connection.'
        );
    }

    $body = curl_exec($ch);
    $error = curl_error($ch);
    $http_status = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

    curl_close($ch);

    if ($body === false) {
        return new WP_Error(
            'backend_connection_error',
            'Backend connection failed: ' . sanitize_text_field($error)
        );
    }

    $data = json_decode($body, true);

    if ($http_status < 200 || $http_status >= 300) {
        $detail = revon_pcb_backend_error_detail(
            $data,
            'HTTP ' . $http_status
        );

        return new WP_Error(
            'backend_http_error',
            'Backend request failed: ' . $detail
        );
    }

    if (!is_array($data)) {
        return new WP_Error(
            'backend_invalid_response',
            'Backend returned an invalid JSON response.'
        );
    }

    return $data;
}

function revon_pcb_backend_get_json($url, $token) {
    $ch = curl_init($url);

    if ($ch === false) {
        return new WP_Error(
            'backend_connection_error',
            'Could not initialize the backend verification connection.'
        );
    }

    $configured = curl_setopt_array($ch, array(
        CURLOPT_HTTPGET => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_CONNECTTIMEOUT => 15,
        CURLOPT_TIMEOUT => 45,
        CURLOPT_HTTPHEADER => array(
            'X-RevOn-Admin-Token: ' . $token,
            'Accept: application/json'
        ),
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_SSL_VERIFYHOST => 2
    ));

    if (!$configured) {
        curl_close($ch);
        return new WP_Error(
            'backend_connection_error',
            'Could not configure the backend verification connection.'
        );
    }

    $body = curl_exec($ch);
    $error = curl_error($ch);
    $status = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

    curl_close($ch);

    if ($body === false) {
        return new WP_Error(
            'backend_connection_error',
            'Backend verification failed: ' . sanitize_text_field($error)
        );
    }

    $data = json_decode($body, true);

    if ($status < 200 || $status >= 300 || !is_array($data)) {
        $detail = revon_pcb_backend_error_detail(
            $data,
            'HTTP ' . $status
        );

        return new WP_Error(
            'backend_verification_error',
            'Backend verification failed: ' . $detail
        );
    }

    return $data;
}

/**
 * Validate the complete FastAPI activation response before changing
 * WordPress registry state.
 */
function revon_pcb_validate_activation_response($response, $expected) {
    if (!is_array($response)) {
        return new WP_Error(
            'invalid_activation_response',
            'Backend activation response is not a valid object.'
        );
    }

    $required_matches = array(
        'status' => 'success',
        'document_id' => $expected['document_id'],
        'logical_document_id' => $expected['logical_document_id'],
        'request_id' => $expected['request_id']
    );

    foreach ($required_matches as $key => $value) {
        if (!isset($response[$key]) || $response[$key] !== $value) {
            return new WP_Error(
                'activation_response_mismatch',
                'Backend activation response mismatch: ' . $key
            );
        }
    }

    if (
        !isset($response['version']) ||
        (int) $response['version'] !== (int) $expected['version']
    ) {
        return new WP_Error(
            'activation_response_mismatch',
            'Backend activation response version mismatch.'
        );
    }

    $expected_previous = $expected['previous_document_id'];
    $actual_previous = $response['previous_document_id'] ?? null;

    if ($actual_previous !== $expected_previous) {
        return new WP_Error(
            'activation_response_mismatch',
            'Backend activation response previous_document_id mismatch.'
        );
    }

    $verification = $response['verification'] ?? null;

    if (!is_array($verification)) {
        return new WP_Error(
            'activation_verification_missing',
            'Backend activation response did not include verification details.'
        );
    }

    $new_ownership = $verification['new_document_ownership'] ?? null;
    $new_active = $verification['new_document_active_state'] ?? null;

    if (
        !is_array($new_ownership) ||
        ($new_ownership['verified'] ?? false) !== true ||
        !is_array($new_active) ||
        ($new_active['verified'] ?? false) !== true
    ) {
        return new WP_Error(
            'activation_verification_failed',
            'Backend did not verify the new document ownership and active state.'
        );
    }

    if ($expected_previous !== null) {
        $previous_ownership = $verification['previous_document_ownership'] ?? null;
        $previous_active = $verification['previous_document_active_state'] ?? null;

        if (
            !is_array($previous_ownership) ||
            ($previous_ownership['verified'] ?? false) !== true ||
            !is_array($previous_active) ||
            ($previous_active['verified'] ?? false) !== true
        ) {
            return new WP_Error(
                'activation_verification_failed',
                'Backend did not verify the previous document ownership and inactive state.'
            );
        }
    }

    return true;
}

function revon_pcb_validate_delete_response($response, $expected) {
    if (!is_array($response)) {
        return new WP_Error('invalid_delete_response', 'Backend delete response is invalid.');
    }

    foreach (array('document_id', 'logical_document_id', 'request_id') as $key) {
        if (!isset($response[$key]) || !isset($expected[$key]) || $response[$key] !== $expected[$key]) {
            return new WP_Error('delete_response_mismatch', 'Backend delete response mismatch: ' . $key);
        }
    }

    if (
        ($response['status'] ?? '') !== 'success' ||
        ($response['verified'] ?? false) !== true ||
        (int) ($response['deleted_vectors'] ?? -1) !== (int) $expected['expected_vectors'] ||
        (int) ($response['remaining_vectors'] ?? -1) !== 0
    ) {
        return new WP_Error('delete_verification_failed', 'Backend did not verify complete vector deletion.');
    }

    return true;
}

/**
 * Validate a read-only absence proof. This is accepted only in the recovery
 * branch when WordPress already has a durable, matching delete operation.
 * It does not claim that this request caused the vectors to be deleted.
 */
function revon_pcb_validate_delete_absence_response($response, $expected) {
    if (!is_array($response)) {
        return new WP_Error('invalid_absence_response', 'Backend absence response is invalid.');
    }

    foreach (array('document_id', 'logical_document_id', 'request_id') as $key) {
        if (!isset($response[$key]) || !isset($expected[$key]) || $response[$key] !== $expected[$key]) {
            return new WP_Error('absence_response_mismatch', 'Backend absence response mismatch: ' . $key);
        }
    }

    if (
        ($response['status'] ?? '') !== 'success' ||
        ($response['outcome'] ?? '') !== 'already_absent' ||
        ($response['verified'] ?? false) !== true ||
        ($response['verified_absence'] ?? false) !== true ||
        (int) ($response['version'] ?? 0) !== (int) $expected['version'] ||
        (int) ($response['deleted_vectors'] ?? -1) !== 0 ||
        (int) ($response['remaining_vectors'] ?? -1) !== 0
    ) {
        return new WP_Error('absence_verification_failed', 'Backend did not verify that all document vectors are absent.');
    }

    return true;
}

function revon_pcb_operation_create($data) {
    global $wpdb;
    $now = current_time('mysql');
    $data['created_at'] = $now;
    $data['updated_at'] = $now;
    return $wpdb->insert(revon_pcb_operations_table(), $data) === 1;
}

function revon_pcb_operation_update($request_id, $data) {
    global $wpdb;

    $allowed_columns = array(
        'status',
        'payload_hash',
        'expected_vectors',
        'attempt_count',
        'last_error',
        'result_json',
        'document_id',
        'previous_document_id',
        'logical_document_id',
        'version',
        'updated_at'
    );

    foreach (array_keys($data) as $column) {
        if (!in_array($column, $allowed_columns, true)) {
            return false;
        }
    }

    $data['updated_at'] = current_time('mysql');
    $table = revon_pcb_operations_table();

    // Confirm the operation exists independently: SELECTing a nullable field
    // cannot distinguish a missing row from a stored SQL NULL.
    $exists = $wpdb->get_var(
        $wpdb->prepare(
            "SELECT request_id FROM {$table} WHERE request_id = %s LIMIT 1",
            $request_id
        )
    );
    if (!$exists) {
        return false;
    }

    $updated = $wpdb->update(
        $table,
        $data,
        array('request_id' => $request_id)
    );

    if ($updated === false) {
        return false;
    }

    // $wpdb->update() may return 0 when values were already identical.
    // Read back each whitelisted field to verify durable persisted state.
    foreach ($data as $column => $expected) {
        $actual = $wpdb->get_var(
            $wpdb->prepare(
                "SELECT `{$column}` FROM {$table} WHERE request_id = %s LIMIT 1",
                $request_id
            )
        );

        if ($expected === null) {
            if ($actual !== null) {
                return false;
            }
        } elseif ((string) $actual !== (string) $expected) {
            return false;
        }
    }

    return true;
}


/* =========================================================
 * 6. Upload and version lifecycle handler
 * ======================================================= */

add_action('admin_post_revon_pcb_upload', 'revon_pcb_handle_upload');

function revon_pcb_handle_upload() {
    if (!current_user_can('manage_options')) wp_die('Unauthorized', 403);
    check_admin_referer('revon_pcb_upload');
    global $wpdb;

    $docs = revon_pcb_table();
    $request_id = wp_generate_uuid4();
    $document_id = wp_generate_uuid4();
    $logical_id = '';
    $previous_id = null;
    $version = 1;
    $stored_path = null;
    $operation_created = false;
    $lock_name = null;
    $lock_acquired = false;
    $file_lock_name = null;
    $file_lock_acquired = false;

    $release = function () use (&$lock_name, &$lock_acquired, &$file_lock_name, &$file_lock_acquired, $wpdb) {
        if ($lock_acquired && $lock_name) {
            $wpdb->get_var($wpdb->prepare('SELECT RELEASE_LOCK(%s)', $lock_name));
            $lock_acquired = false;
        }
        if ($file_lock_acquired && $file_lock_name) {
            $wpdb->get_var($wpdb->prepare('SELECT RELEASE_LOCK(%s)', $file_lock_name));
            $file_lock_acquired = false;
        }
    };
    $fail = function ($message, $preserve = false) use (&$stored_path, &$operation_created, &$request_id, $release) {
        if (!$preserve && $stored_path && file_exists($stored_path)) @unlink($stored_path);
        if ($operation_created) revon_pcb_operation_update($request_id, array(
            'status' => $preserve ? 'reconciliation_required' : 'failed',
            'last_error' => sanitize_text_field($message)
        ));
        $release();
        revon_pcb_redirect($message, 'error');
    };

    if (empty($_FILES['policy_pdf'])) $fail('Choose a PDF file.');
    $file = $_FILES['policy_pdf'];
    if (!isset($file['error'],$file['tmp_name'],$file['size'],$file['name']) ||
        (int)$file['error'] !== UPLOAD_ERR_OK || !is_uploaded_file($file['tmp_name'])) $fail('Upload failed. Select the PDF again.');
    if ((int)$file['size'] < 1 || (int)$file['size'] > 10485760) $fail('PDF must be 10 MB or smaller.');
    if (strtolower(pathinfo($file['name'], PATHINFO_EXTENSION)) !== 'pdf') $fail('Only PDF files are allowed.');

    $fh=fopen($file['tmp_name'],'rb'); $signature=$fh ? fread($fh,5) : ''; if ($fh) fclose($fh);
    if ($signature !== '%PDF-') $fail('Invalid PDF signature.');
    $hash = hash_file('sha256', $file['tmp_name']);
    if (!$hash || !preg_match('/^[a-f0-9]{64}$/i', $hash)) {
        $fail('Could not calculate a valid PDF SHA-256 hash.');
    }

    // Serialize identical-file checks across different policy IDs. The lock
    // is held until the registry row is inserted, making the duplicate check
    // and registration one protected critical section.
    $file_lock_name = 'revon_file_' . md5($hash);
    $file_lock_result = $wpdb->get_var(
        $wpdb->prepare('SELECT GET_LOCK(%s, 15)', $file_lock_name)
    );
    if ((string) $file_lock_result !== '1') {
        $fail('Could not acquire the duplicate-file protection lock.');
    }
    $file_lock_acquired = true;

    $duplicate = $wpdb->get_var(
        $wpdb->prepare("SELECT document_id FROM {$docs} WHERE file_hash = %s LIMIT 1", $hash)
    );
    if ($duplicate) {
        $fail('This exact PDF is already registered. Document ID: ' . $duplicate);
    }

    $type=isset($_POST['upload_type']) ? sanitize_key(wp_unslash($_POST['upload_type'])) : '';
    if (!in_array($type,array('new_policy','new_version'),true)) $fail('Invalid upload type.');

    if ($type === 'new_policy') {
        $logical_id = wp_generate_uuid4();
    } else {
        $logical_id = isset($_POST['logical_document_id'])
            ? sanitize_text_field(wp_unslash($_POST['logical_document_id']))
            : '';
        if (!$logical_id) {
            $fail('Select an existing policy.');
        }
    }

    // Every upload, including a brand-new policy, must hold the same
    // lifecycle lock used by deletion. This prevents deleting a document
    // while its upload/index/activation workflow is still running.
    $lock_name = 'revon_policy_' . md5($logical_id);
    $got = $wpdb->get_var($wpdb->prepare('SELECT GET_LOCK(%s,15)', $lock_name));
    if ((string) $got !== '1') {
        $fail('Could not acquire policy lifecycle lock.');
    }
    $lock_acquired = true;

    if ($type === 'new_version') {
        $rows = $wpdb->get_results(
            $wpdb->prepare(
                "SELECT document_id,version,status FROM {$docs} WHERE logical_document_id=%s ORDER BY version DESC",
                $logical_id
            ),
            ARRAY_A
        );

        if (!$rows) {
            $fail('Selected policy was not found.');
        }

        $active = array_values(array_filter($rows, function ($row) {
            return $row['status'] === 'active';
        }));

        if (count($active) !== 1) {
            $fail('Policy must have exactly one active version. Reconcile registry first.');
        }

        $version = (int) $rows[0]['version'] + 1;
        $previous_id = $active[0]['document_id'];
    }

    $backend=revon_pcb_backend_config();
    if (is_wp_error($backend)) $fail($backend->get_error_message());
    $dir=revon_pcb_private_dir();
    if (!is_dir($dir) && !wp_mkdir_p($dir)) $fail('Could not create private storage.');
    if (!is_dir($dir) || !is_writable($dir)) $fail('Private storage is not writable.');
    @file_put_contents($dir.'/.htaccess',"Require all denied\nDeny from all\n");
    @file_put_contents($dir.'/index.php',"<?php\nhttp_response_code(404); exit;\n");
    @file_put_contents($dir . '/web.config', '<?xml version="1.0"?><configuration><system.webServer><authorization><deny users="*" /></authorization></system.webServer></configuration>');
    $stored_path=trailingslashit($dir).wp_generate_uuid4().'.pdf';
    if (!move_uploaded_file($file['tmp_name'],$stored_path)) $fail('Could not save PDF privately.');

    $operation_created=revon_pcb_operation_create(array(
        'request_id'=>$request_id,'operation_type'=>'upload','document_id'=>$document_id,
        'previous_document_id'=>$previous_id,'logical_document_id'=>$logical_id,'version'=>$version,
        'status'=>'pending','payload_hash'=>hash('sha256',$hash.'|'.$logical_id.'|'.$version.'|'.$document_id),
        'expected_vectors'=>null,'attempt_count'=>0,'last_error'=>null,'result_json'=>null
    ));
    if (!$operation_created) $fail('Could not create durable operation record.');

    $inserted=$wpdb->insert($docs,array(
        'document_id'=>$document_id,'logical_document_id'=>$logical_id,'filename'=>sanitize_file_name($file['name']),
        'file_hash'=>$hash,'version'=>$version,'storage_path'=>$stored_path,'total_pages'=>0,'total_chunks'=>0,
        'status'=>'processing','created_at'=>current_time('mysql'),'updated_at'=>current_time('mysql')
    ),array('%s','%s','%s','%s','%d','%s','%d','%d','%s','%s','%s'));
    if ($inserted !== 1) {
        $fail('Registry insert failed. Preserve operation and reconcile.', true);
    }

    // The unique file hash is now visible in the durable registry; release
    // only the hash lock while retaining the policy lifecycle lock.
    if ($file_lock_acquired && $file_lock_name) {
        $wpdb->get_var($wpdb->prepare('SELECT RELEASE_LOCK(%s)', $file_lock_name));
        $file_lock_acquired = false;
    }

    if (!revon_pcb_operation_update($request_id, array(
        'status' => 'uploading',
        'attempt_count' => 1
    ))) {
        $fail('Could not update the operation ledger before backend upload. Preserve and reconcile.', true);
    }

    $fields=array(
        'file'=>new CURLFile($stored_path,'application/pdf',sanitize_file_name($file['name'])),
        'upload_type'=>$type,'logical_document_id'=>$logical_id,'document_id'=>$document_id,
        'request_id'=>$request_id,'version'=>(string)$version
    );
    $uploaded=revon_pcb_backend_post_form($backend['base'].'/api/v1/documents/upload',$backend['token'],$fields);
    if (is_wp_error($uploaded)) $fail('Upload outcome is unknown; PDF and operation retained. Verify status before retry. '.$uploaded->get_error_message(),true);

    if (($uploaded['status']??'')!=='success' || ($uploaded['document_id']??'')!==$document_id ||
        ($uploaded['logical_document_id']??'')!==$logical_id || ($uploaded['request_id']??'')!==$request_id ||
        (int)($uploaded['version']??0)!==$version || ($uploaded['is_active']??null)!==false) {
        $fail('Backend response does not match operation identifiers. Preserve and reconcile.',true);
    }

    $expected=absint($uploaded['expected_vectors']??$uploaded['total_chunks']??0);
    if ($expected<1) $fail('Backend returned invalid expected vector count. Preserve and reconcile.',true);
    $registry_metadata_updated = $wpdb->update(
        $docs,
        array(
            'total_pages' => absint($uploaded['total_pages'] ?? 0),
            'total_chunks' => $expected,
            'updated_at' => current_time('mysql')
        ),
        array('document_id' => $document_id),
        array('%d', '%d', '%s'),
        array('%s')
    );

    if ($registry_metadata_updated === false) {
        $fail('Backend indexing succeeded, but WordPress could not update document metadata. Preserve and reconcile.', true);
    }

    if (!revon_pcb_operation_update($request_id, array(
        'status' => 'indexed',
        'expected_vectors' => $expected,
        'result_json' => wp_json_encode($uploaded)
    ))) {
        $fail('Could not update the operation ledger after indexing. Preserve and reconcile.', true);
    }

    $status_url=add_query_arg(array('expected_vectors'=>$expected,'request_id'=>$request_id,'logical_document_id'=>$logical_id),
        $backend['base'].'/api/v1/documents/'.rawurlencode($document_id).'/status');
    $verified=revon_pcb_backend_get_json($status_url,$backend['token']);
    if (
        is_wp_error($verified) ||
        ($verified['verified'] ?? false) !== true ||
        ($verified['ownership_verified'] ?? false) !== true
    ) {
        $why = is_wp_error($verified)
            ? $verified->get_error_message()
            : 'Pinecone vector count and ownership verification did not both succeed.';
        $fail('Indexed but not verified. Preserve and reconcile. ' . $why, true);
    }
    if (!revon_pcb_operation_update($request_id, array('status' => 'verified'))) {
        $fail('Could not update the operation ledger after Pinecone verification. Preserve and reconcile.', true);
    }

    $activate = array(
        'document_id' => $document_id,
        'logical_document_id' => $logical_id,
        'version' => (string) $version,
        'expected_vectors' => (string) $expected,
        'request_id' => $request_id
    );
    if ($previous_id) {
        $activate['previous_document_id'] = $previous_id;
    }

    if (!revon_pcb_operation_update($request_id, array('status' => 'activating'))) {
        $fail('Could not mark the operation as activating. Preserve and reconcile.', true);
    }
    $activated=revon_pcb_backend_post_form($backend['base'].'/api/v1/documents/activate',$backend['token'],$activate);
    if (is_wp_error($activated)) {
        $fail(
            'Activation outcome is unknown/failed. Preserve records and reconcile before retry. ' .
            $activated->get_error_message(),
            true
        );
    }

    $activation_validation = revon_pcb_validate_activation_response(
        $activated,
        array(
            'document_id' => $document_id,
            'logical_document_id' => $logical_id,
            'version' => $version,
            'request_id' => $request_id,
            'previous_document_id' => $previous_id
        )
    );

    if (is_wp_error($activation_validation)) {
        $fail(
            'Backend activation response could not be fully verified. Preserve records and reconcile. ' .
            $activation_validation->get_error_message(),
            true
        );
    }

    if ($wpdb->query('START TRANSACTION') === false) {
        revon_pcb_operation_update($request_id, array(
            'status' => 'reconciliation_required',
            'last_error' => 'Could not start WordPress registry transaction after Pinecone activation.'
        ));
        $fail(
            'Pinecone activation succeeded, but WordPress could not start its registry transaction. Manual reconciliation is required.',
            true
        );
    }
    $ok=true;
    if ($previous_id) {
        $old=$wpdb->update($docs,array('status'=>'inactive','updated_at'=>current_time('mysql')),
            array('document_id'=>$previous_id,'status'=>'active'),array('%s','%s'),array('%s','%s'));
        if ($old!==1) $ok=false;
    }
    if ($ok) {
        $new=$wpdb->update($docs,array('status'=>'active','updated_at'=>current_time('mysql')),
            array('document_id'=>$document_id,'status'=>'processing'),array('%s','%s'),array('%s','%s'));
        if ($new!==1) $ok=false;
    }
    if ($ok) {
        if ($wpdb->query('COMMIT') === false) {
            $wpdb->query('ROLLBACK');
            revon_pcb_operation_update($request_id, array(
                'status' => 'reconciliation_required',
                'last_error' => 'WordPress registry COMMIT failed after Pinecone activation.'
            ));
            $fail('Pinecone activation succeeded, but WordPress registry COMMIT failed. Manual reconciliation is required.', true);
        }

        if (!revon_pcb_operation_update($request_id, array(
            'status' => 'completed',
            'last_error' => null
        ))) {
            $fail('Registry activation committed, but the operation ledger could not be marked completed. Manual reconciliation is required.', true);
        }
    } else {
        $wpdb->query('ROLLBACK');
        revon_pcb_operation_update($request_id,array('status'=>'reconciliation_required',
            'last_error'=>'Pinecone activation succeeded but WordPress registry transaction failed.'));
        $fail('Pinecone activated, but WordPress transaction failed. Manual reconciliation required.',true);
    }

    $stored_path=null; // Keep original private PDF as durable source.
    $release();
    revon_pcb_redirect(sprintf('Policy version %d indexed, verified and activated. Document ID: %s',$version,$document_id),'success');
}
/* =========================================================
 * 7. Verified document deletion lifecycle
 * ======================================================= */

add_action('admin_post_revon_pcb_deactivate_document', 'revon_pcb_handle_deactivate_document');

function revon_pcb_validate_deactivation_response($response, $expected) {
    if (!is_array($response)) {
        return new WP_Error('invalid_deactivation_response', 'Backend deactivation response is not a valid object.');
    }

    $matches = array(
        'status' => 'success',
        'document_id' => $expected['document_id'],
        'logical_document_id' => $expected['logical_document_id'],
        'request_id' => $expected['request_id'],
        'upload_request_id' => $expected['upload_request_id'],
    );
    foreach ($matches as $key => $value) {
        if (!isset($response[$key]) || $response[$key] !== $value) {
            return new WP_Error('deactivation_response_mismatch', 'Backend deactivation response mismatch: ' . $key);
        }
    }

    if (
        ($response['is_active'] ?? null) !== false ||
        ($response['verified'] ?? false) !== true ||
        (int) ($response['total_updated'] ?? -1) < 0
    ) {
        return new WP_Error('deactivation_verification_failed', 'Backend did not confirm the inactive vector state.');
    }

    return true;
}

function revon_pcb_handle_deactivate_document() {
    if (!current_user_can('manage_options')) {
        wp_die('Unauthorized', 403);
    }

    $document_id = isset($_POST['document_id'])
        ? sanitize_text_field(wp_unslash($_POST['document_id']))
        : '';
    if (!$document_id || !preg_match('/^[a-f0-9-]{36}$/i', $document_id)) {
        revon_pcb_redirect('A valid document ID is required.', 'error');
    }
    check_admin_referer('revon_pcb_deactivate_' . $document_id);

    global $wpdb;
    $docs = revon_pcb_table();
    $ops = revon_pcb_operations_table();
    $document = $wpdb->get_row(
        $wpdb->prepare("SELECT * FROM {$docs} WHERE document_id = %s LIMIT 1", $document_id),
        ARRAY_A
    );
    if (!$document || $document['status'] !== 'active') {
        revon_pcb_redirect('Only an active registry document can be deactivated. No changes were made.', 'error');
    }

    $logical_id = (string) $document['logical_document_id'];
    $version = (int) $document['version'];
    $expected_vectors = (int) $document['total_chunks'];
    if ($logical_id === '' || $version < 1 || $expected_vectors < 1) {
        revon_pcb_redirect('Document metadata is incomplete. Deactivation stopped.', 'error');
    }

    $lock_name = 'revon_policy_' . md5($logical_id);
    $lock = $wpdb->get_var($wpdb->prepare('SELECT GET_LOCK(%s, 15)', $lock_name));
    if ((string) $lock !== '1') {
        revon_pcb_redirect('Could not acquire the policy lifecycle lock. Try again later.', 'error');
    }
    $release = function () use ($wpdb, $lock_name) {
        $wpdb->get_var($wpdb->prepare('SELECT RELEASE_LOCK(%s)', $lock_name));
    };

    $document = $wpdb->get_row(
        $wpdb->prepare("SELECT * FROM {$docs} WHERE document_id = %s LIMIT 1", $document_id),
        ARRAY_A
    );
    if (!$document || $document['status'] !== 'active') {
        $release();
        revon_pcb_redirect('Document state changed or is no longer active. Deactivation stopped.', 'error');
    }

    $upload_row = $wpdb->get_row(
        $wpdb->prepare(
            "SELECT request_id, status FROM {$ops} WHERE operation_type = 'upload' AND document_id = %s ORDER BY id DESC LIMIT 1",
            $document_id
        ),
        ARRAY_A
    );
    if (!$upload_row || $upload_row['status'] !== 'completed') {
        $release();
        revon_pcb_redirect('A completed original upload audit record is required. Deactivation stopped.', 'error');
    }

    $backend = revon_pcb_backend_config();
    if (is_wp_error($backend)) {
        $release();
        revon_pcb_redirect($backend->get_error_message(), 'error');
    }

    $request_id = wp_generate_uuid4();
    $payload_hash = hash('sha256', implode('|', array(
        $document_id, $logical_id, (string) $version,
        (string) $expected_vectors, (string) $upload_row['request_id']
    )));
    $created = revon_pcb_operation_create(array(
        'request_id' => $request_id,
        'operation_type' => 'deactivate',
        'document_id' => $document_id,
        'previous_document_id' => null,
        'logical_document_id' => $logical_id,
        'version' => $version,
        'status' => 'pending',
        'payload_hash' => $payload_hash,
        'expected_vectors' => $expected_vectors,
        'attempt_count' => 0,
        'last_error' => null,
        'result_json' => null,
    ));
    if (!$created) {
        $release();
        revon_pcb_redirect('Could not create the durable deactivation operation. No backend changes were made.', 'error');
    }

    if (!revon_pcb_operation_update($request_id, array('status' => 'deactivating'))) {
        $release();
        revon_pcb_redirect('Could not persist deactivation intent. No backend changes were made.', 'error');
    }

    $response = revon_pcb_backend_post_form(
        $backend['base'] . '/api/v1/documents/deactivate',
        $backend['token'],
        array(
            'document_id' => $document_id,
            'logical_document_id' => $logical_id,
            'version' => $version,
            'expected_vectors' => $expected_vectors,
            'request_id' => $request_id,
            'upload_request_id' => $upload_row['request_id'],
        )
    );

    if (is_wp_error($response)) {
        revon_pcb_operation_update($request_id, array(
            'status' => 'reconciliation_required',
            'last_error' => $response->get_error_message(),
        ));
        $release();
        revon_pcb_redirect('Deactivation outcome is unknown. PDF and registry were retained; reconcile before retrying. ' . $response->get_error_message(), 'error');
    }

    $expected = array(
        'document_id' => $document_id,
        'logical_document_id' => $logical_id,
        'request_id' => $request_id,
        'upload_request_id' => $upload_row['request_id'],
    );
    $valid = revon_pcb_validate_deactivation_response($response, $expected);
    if (is_wp_error($valid)) {
        revon_pcb_operation_update($request_id, array(
            'status' => 'reconciliation_required',
            'last_error' => $valid->get_error_message(),
            'result_json' => wp_json_encode($response),
        ));
        $release();
        revon_pcb_redirect('Backend response could not be verified. PDF and registry were retained. ' . $valid->get_error_message(), 'error');
    }

    $wpdb->query('START TRANSACTION');
    $updated = $wpdb->update(
        $docs,
        array('status' => 'inactive', 'updated_at' => current_time('mysql')),
        array('document_id' => $document_id, 'status' => 'active'),
        array('%s', '%s'),
        array('%s', '%s')
    );
    $stored_status = $wpdb->get_var(
        $wpdb->prepare("SELECT status FROM {$docs} WHERE document_id = %s LIMIT 1", $document_id)
    );
    if ($updated === false || $stored_status !== 'inactive') {
        $wpdb->query('ROLLBACK');
        revon_pcb_operation_update($request_id, array(
            'status' => 'reconciliation_required',
            'last_error' => 'Pinecone is verified inactive, but WordPress registry update failed.',
            'result_json' => wp_json_encode($response),
        ));
        $release();
        revon_pcb_redirect('Pinecone is inactive but WordPress registry reconciliation is required. No files or vectors were deleted.', 'error');
    }

    $result_saved = revon_pcb_operation_update($request_id, array(
        'status' => 'completed',
        'result_json' => wp_json_encode($response),
        'last_error' => null,
    ));
    if (!$result_saved) {
        $wpdb->query('ROLLBACK');
        revon_pcb_operation_update($request_id, array(
            'status' => 'reconciliation_required',
            'last_error' => 'Could not persist the completed deactivation result.',
            'result_json' => wp_json_encode($response),
        ));
        $release();
        revon_pcb_redirect('Deactivation result persistence failed. Manual reconciliation is required.', 'error');
    }

    $wpdb->query('COMMIT');
    $release();
    revon_pcb_redirect('Policy document deactivated and Pinecone inactive state verified. PDF and vectors were retained.', 'success');
}


add_action('admin_post_revon_pcb_delete_document', 'revon_pcb_handle_delete_document');

function revon_pcb_handle_delete_document() {
    if (!current_user_can('manage_options')) {
        wp_die('Unauthorized', 403);
    }

    $document_id = isset($_POST['document_id'])
        ? sanitize_text_field(wp_unslash($_POST['document_id']))
        : '';

    if (!$document_id || !preg_match('/^[a-f0-9-]{36}$/i', $document_id)) {
        revon_pcb_redirect('A valid document ID is required.', 'error');
    }

    check_admin_referer('revon_pcb_delete_' . $document_id);

    global $wpdb;
    $docs = revon_pcb_table();
    $ops = revon_pcb_operations_table();

    $document = $wpdb->get_row(
        $wpdb->prepare("SELECT * FROM {$docs} WHERE document_id = %s LIMIT 1", $document_id),
        ARRAY_A
    );
    if (!$document) {
        revon_pcb_redirect('Document not found in the registry.', 'error');
    }
    $deletable_statuses = array('inactive', 'failed');
    if (!in_array($document['status'], $deletable_statuses, true)) {
        revon_pcb_redirect(
            'Deletion is allowed only for inactive or failed documents. Current state: ' . $document['status'] . '. Reconcile in-progress states first.',
            'error'
        );
    }

    $logical_id = (string) $document['logical_document_id'];
    $version = (int) $document['version'];
    $expected_vectors = (int) $document['total_chunks'];
    if ($logical_id === '' || $version < 1 || $expected_vectors < 1) {
        revon_pcb_redirect('Document metadata is incomplete. Reconcile it before deletion.', 'error');
    }

    // Use the same per-policy lock namespace as the upload/version workflow.
    $lock_name = 'revon_policy_' . md5($logical_id);
    $lock = $wpdb->get_var($wpdb->prepare('SELECT GET_LOCK(%s, 15)', $lock_name));
    if ((string) $lock !== '1') {
        revon_pcb_redirect('Could not acquire the policy lifecycle lock. Try again later.', 'error');
    }
    $release_lock = function () use ($wpdb, $lock_name) {
        $wpdb->get_var($wpdb->prepare('SELECT RELEASE_LOCK(%s)', $lock_name));
    };

    // Re-read the registry state after obtaining the lock.
    $document = $wpdb->get_row(
        $wpdb->prepare("SELECT * FROM {$docs} WHERE document_id = %s LIMIT 1", $document_id),
        ARRAY_A
    );
    if (
        !$document ||
        !in_array($document['status'], array('inactive', 'failed'), true)
    ) {
        $release_lock();
        revon_pcb_redirect(
            'Document state changed or is not in a deletable terminal state. Deletion stopped.',
            'error'
        );
    }

    // The backend delete API verifies the original upload request identity.
    $original_request_id = $wpdb->get_var(
        $wpdb->prepare(
            "SELECT request_id FROM {$ops} WHERE operation_type = 'upload' AND document_id = %s ORDER BY id DESC LIMIT 1",
            $document_id
        )
    );
    if (!$original_request_id) {
        $release_lock();
        revon_pcb_redirect('Original upload audit record is missing. Deletion stopped.', 'error');
    }

    $original_upload_status = $wpdb->get_var(
        $wpdb->prepare(
            "SELECT status FROM {$ops} WHERE request_id = %s AND operation_type = 'upload' AND document_id = %s LIMIT 1",
            $original_request_id,
            $document_id
        )
    );

    if (!in_array($original_upload_status, array('completed', 'failed'), true)) {
        $release_lock();
        revon_pcb_redirect(
            'Original upload operation is not in a terminal state. Reconcile it before deletion.',
            'error'
        );
    }

    $backend = revon_pcb_backend_config();
    if (is_wp_error($backend)) {
        $release_lock();
        revon_pcb_redirect($backend->get_error_message(), 'error');
    }

    // Resolve and validate the private file path. A missing file is allowed only
    // when a previous deletion operation can be reconciled to verified vector absence.
    $private_dir = realpath(revon_pcb_private_dir());
    $stored_path = (string) $document['storage_path'];
    $real_file = is_file($stored_path) ? realpath($stored_path) : false;
    if (!$private_dir || ($real_file && dirname($real_file) !== $private_dir)) {
        $release_lock();
        revon_pcb_redirect('Private PDF path failed its safety check. No deletion performed.', 'error');
    }

    // Resume only from a previously stored, fully verified backend deletion result.
    $prior_delete = $wpdb->get_row(
        $wpdb->prepare(
            "SELECT * FROM {$ops} WHERE operation_type = 'delete' AND document_id = %s ORDER BY id DESC LIMIT 1",
            $document_id
        ),
        ARRAY_A
    );
    $backend_response = null;
    $delete_operation_id = null;
    $already_verified = false;

    if ($prior_delete && !empty($prior_delete['result_json'])) {
        $prior_response = json_decode($prior_delete['result_json'], true);
        $prior_validation = revon_pcb_validate_delete_response($prior_response, array(
            'document_id' => $document_id,
            'logical_document_id' => $logical_id,
            'request_id' => $original_request_id,
            'expected_vectors' => $expected_vectors
        ));
        if (!is_wp_error($prior_validation)) {
            $backend_response = $prior_response;
            $delete_operation_id = $prior_delete['request_id'];
            $already_verified = true;
        }
    }

    // Recover an ambiguous prior attempt using a read-only backend check.
    // Require the durable WP operation to match this exact document/version
    // and the original upload request before trusting an absence response.
    if ($prior_delete && !$already_verified) {
        $expected_delete_hash = hash(
            'sha256',
            $document_id . '|' . $logical_id . '|' . $version . '|' . $original_request_id
        );
        $recoverable_statuses = array('deleting', 'reconciliation_required', 'vectors_deleted', 'vectors_absent_verified');

        if (
            !in_array($prior_delete['status'], $recoverable_statuses, true) ||
            (string) $prior_delete['logical_document_id'] !== $logical_id ||
            (int) $prior_delete['version'] !== $version ||
            (int) $prior_delete['expected_vectors'] !== $expected_vectors ||
            !hash_equals($expected_delete_hash, (string) $prior_delete['payload_hash'])
        ) {
            $release_lock();
            revon_pcb_redirect(
                'The previous deletion ledger does not match this document lifecycle. No cleanup or new delete request was performed.',
                'error'
            );
        }

        $absence_response = revon_pcb_backend_post_form(
            $backend['base'] . '/api/v1/documents/delete/verify-absence',
            $backend['token'],
            array(
                'document_id' => $document_id,
                'logical_document_id' => $logical_id,
                'version' => (string) $version,
                'expected_vectors' => (string) $expected_vectors,
                'request_id' => $original_request_id
            )
        );

        if (is_wp_error($absence_response)) {
            revon_pcb_operation_update($prior_delete['request_id'], array(
                'status' => 'reconciliation_required',
                'last_error' => 'Read-only vector absence verification failed: ' . $absence_response->get_error_message()
            ));
            $release_lock();
            revon_pcb_redirect(
                'The previous deletion outcome remains unresolved. PDF and registry were preserved; no second delete request was sent.',
                'error'
            );
        }

        $absence_validation = revon_pcb_validate_delete_absence_response(
            $absence_response,
            array(
                'document_id' => $document_id,
                'logical_document_id' => $logical_id,
                'request_id' => $original_request_id,
                'version' => $version
            )
        );

        if (is_wp_error($absence_validation)) {
            revon_pcb_operation_update($prior_delete['request_id'], array(
                'status' => 'reconciliation_required',
                'result_json' => wp_json_encode($absence_response),
                'last_error' => $absence_validation->get_error_message()
            ));
            $release_lock();
            revon_pcb_redirect(
                'Backend could not prove vector absence. PDF and registry were preserved; manual reconciliation is required.',
                'error'
            );
        }

        // Persist the proof before removing the private PDF or registry row.
        if (!revon_pcb_operation_update($prior_delete['request_id'], array(
            'status' => 'vectors_absent_verified',
            'result_json' => wp_json_encode($absence_response),
            'last_error' => null
        ))) {
            $release_lock();
            revon_pcb_redirect(
                'Vectors appear absent, but the reconciliation proof could not be saved. PDF and registry were preserved.',
                'error'
            );
        }

        $backend_response = $absence_response;
        $delete_operation_id = $prior_delete['request_id'];
        $already_verified = true;
    }

    if (!$already_verified) {
        if (!$real_file || !is_readable($real_file)) {
            $release_lock();
            revon_pcb_redirect('Private PDF is missing and no verified deletion record exists. Manual reconciliation required.', 'error');
        }

        $delete_operation_id = wp_generate_uuid4();
        $created = revon_pcb_operation_create(array(
            'request_id' => $delete_operation_id,
            'operation_type' => 'delete',
            'document_id' => $document_id,
            'previous_document_id' => null,
            'logical_document_id' => $logical_id,
            'version' => $version,
            'status' => 'pending',
            'payload_hash' => hash('sha256', $document_id . '|' . $logical_id . '|' . $version . '|' . $original_request_id),
            'expected_vectors' => $expected_vectors,
            'attempt_count' => 0,
            'last_error' => null,
            'result_json' => null
        ));
        if (!$created) {
            $release_lock();
            revon_pcb_redirect('Could not create the durable deletion audit record.', 'error');
        }

        if (!revon_pcb_operation_update($delete_operation_id, array(
            'status' => 'deleting',
            'attempt_count' => 1
        ))) {
            $release_lock();
            revon_pcb_redirect(
                'Could not persist the deletion operation state. Backend was not called; preserve and reconcile.',
                'error'
            );
        }

        // IMPORTANT: request_id here is the original upload request ID,
        // because Pinecone metadata ownership is tied to that ID.
        $backend_response = revon_pcb_backend_post_form(
            $backend['base'] . '/api/v1/documents/delete',
            $backend['token'],
            array(
                'document_id' => $document_id,
                'logical_document_id' => $logical_id,
                'version' => (string) $version,
                'expected_vectors' => (string) $expected_vectors,
                'request_id' => $original_request_id
            )
        );

        if (is_wp_error($backend_response)) {
            revon_pcb_operation_update($delete_operation_id, array(
                'status' => 'reconciliation_required',
                'last_error' => $backend_response->get_error_message()
            ));
            $release_lock();
            revon_pcb_redirect('Backend deletion outcome is unknown. PDF and registry were preserved; reconcile before retry.', 'error');
        }

        $validation = revon_pcb_validate_delete_response($backend_response, array(
            'document_id' => $document_id,
            'logical_document_id' => $logical_id,
            'request_id' => $original_request_id,
            'expected_vectors' => $expected_vectors
        ));
        if (is_wp_error($validation)) {
            revon_pcb_operation_update($delete_operation_id, array(
                'status' => 'reconciliation_required',
                'result_json' => wp_json_encode($backend_response),
                'last_error' => $validation->get_error_message()
            ));
            $release_lock();
            revon_pcb_redirect('Backend deletion could not be verified. PDF and registry were preserved.', 'error');
        }

        if (!revon_pcb_operation_update($delete_operation_id, array(
            'status' => 'vectors_deleted',
            'result_json' => wp_json_encode($backend_response),
            'last_error' => null
        ))) {
            $release_lock();
            revon_pcb_redirect('Vectors were deleted, but the audit record update failed. Manual reconciliation required.', 'error');
        }
    }

    // At this point a durable audit result confirms that the backend verified zero remaining vectors.
    if ($real_file && !unlink($real_file)) {
        revon_pcb_operation_update($delete_operation_id, array(
            'status' => 'reconciliation_required',
            'last_error' => 'Backend vectors were verified absent, but private PDF removal failed.'
        ));
        $release_lock();
        revon_pcb_redirect('Pinecone vectors are confirmed absent, but private PDF cleanup failed. Retry cleanup after checking file permissions.', 'error');
    }

    if ($wpdb->query('START TRANSACTION') === false) {
        revon_pcb_operation_update($delete_operation_id, array(
            'status' => 'reconciliation_required',
            'last_error' => 'Vectors and private PDF were deleted, but the registry transaction could not start.'
        ));
        $release_lock();
        revon_pcb_redirect('Vectors and PDF were deleted, but registry cleanup needs reconciliation.', 'error');
    }

    $deleted_rows = $wpdb->delete(
        $docs,
        array('document_id' => $document_id, 'status' => $document['status']),
        array('%s', '%s')
    );

    if ($deleted_rows !== 1 || $wpdb->query('COMMIT') === false) {
        $wpdb->query('ROLLBACK');
        revon_pcb_operation_update($delete_operation_id, array(
            'status' => 'reconciliation_required',
            'last_error' => 'Vectors and private PDF were deleted, but registry row deletion failed.'
        ));
        $release_lock();
        revon_pcb_redirect('Vectors and PDF were deleted, but registry cleanup failed. Manual reconciliation required.', 'error');
    }

    $audit_updated = revon_pcb_operation_update($delete_operation_id, array(
        'status' => 'completed',
        'last_error' => null,
        'result_json' => wp_json_encode($backend_response)
    ));

    $release_lock();

    if (!$audit_updated) {
        revon_pcb_redirect('Document was deleted, but the audit completion update failed. Review the operation ledger.', 'error');
    }

    revon_pcb_redirect(
        (($backend_response['outcome'] ?? '') === 'already_absent')
            ? 'Document and private PDF removed; Pinecone vectors were already absent and this was verified.'
            : 'Document, private PDF and verified Pinecone vectors deleted successfully.',
        'success'
    );
}


/* =========================================================
 * 8. Public chat REST proxy, SSE streaming, and visitor UI
 * ======================================================= */

add_action('rest_api_init', function () {
    register_rest_route('revon/v1', '/chat', array(
        'methods' => WP_REST_Server::CREATABLE,
        'callback' => 'revon_pcb_public_chat_proxy',
        // Deliberately public: visitors do not need WordPress accounts.
        // Server-to-server authentication is applied to the FastAPI request.
        'permission_callback' => '__return_true',
        'args' => array(
            'question' => array(
                'required' => true,
                'type' => 'string',
                'validate_callback' => function ($value) {
                    return is_string($value);
                }
            ),
            'conversation_id' => array(
                'required' => false,
                'type' => 'string',
                'validate_callback' => function ($value) {
                    return is_string($value) && strlen($value) <= 100;
                }
            )
        )
    ));

    register_rest_route('revon/v1', '/chat/reset', array(
        'methods' => WP_REST_Server::CREATABLE,
        'callback' => 'revon_pcb_public_chat_reset_proxy',
        'permission_callback' => '__return_true',
        'args' => array(
            'conversation_id' => array(
                'required' => true,
                'type' => 'string',
                'validate_callback' => function ($value) {
                    return is_string($value) && strlen($value) <= 100;
                }
            )
        )
    ));
});

function revon_pcb_public_chat_rate_limit_key() {
    // Do not trust browser-supplied X-Forwarded-For / visitor-IP headers.
    // REMOTE_ADDR is the immediate peer; if WordPress is behind a proxy,
    // configure trusted-proxy handling at the hosting layer before launch.
    $remote = isset($_SERVER['REMOTE_ADDR']) ? (string) $_SERVER['REMOTE_ADDR'] : '';
    if (!$remote || !filter_var($remote, FILTER_VALIDATE_IP)) {
        $remote = 'unknown';
    }
    return 'revon_pcb_chat_' . hash('sha256', $remote . '|' . (string) (int) floor(time() / 60));
}

function revon_pcb_public_chat_proxy(WP_REST_Request $request) {
    $question = $request->get_param('question');
    if (!is_string($question)) {
        return new WP_Error('invalid_question', 'Please enter a question.', array('status' => 400));
    }

    $question = trim(sanitize_textarea_field($question));
    if ($question === '') {
        return new WP_Error('empty_question', 'Please enter a question.', array('status' => 400));
    }

    $length = function_exists('mb_strlen') ? mb_strlen($question, 'UTF-8') : strlen($question);
    if ($length > 2000) {
        return new WP_Error('question_too_long', 'Your question must be 2,000 characters or fewer.', array('status' => 400));
    }

    $conversation_id = $request->get_param('conversation_id');
    if (is_string($conversation_id)) {
        $conversation_id = sanitize_text_field(trim($conversation_id));
        if ($conversation_id === '') {
            $conversation_id = null;
        }
    } else {
        $conversation_id = null;
    }

    // Per-immediate-client-IP minute bucket. This is a basic application-layer
    // guard; CDN/WAF rate limiting and atomic shared counters are still needed
    // for robust protection on a multi-server WordPress deployment.
    $rate_key = revon_pcb_public_chat_rate_limit_key();
    $count = (int) get_transient($rate_key);
    if ($count >= 15) {
        return new WP_Error('chat_rate_limited', 'Too many questions. Please wait a minute and try again.', array('status' => 429));
    }
    set_transient($rate_key, $count + 1, 70);

    $base = untrailingslashit((string) get_option('revon_pcb_api_base_url', ''));
    $token = (string) get_option('revon_pcb_chat_token', '');
    if (!$base || !wp_http_validate_url($base) || strpos($base, 'https://') !== 0 || !$token) {
        return new WP_Error('chat_not_configured', 'The chatbot is temporarily unavailable. Please try again later.', array('status' => 503));
    }

    $visitor_ip = isset($_SERVER['REMOTE_ADDR']) ? (string) $_SERVER['REMOTE_ADDR'] : 'unknown';

    $body_params = array('question' => $question);
    if ($conversation_id) {
        $body_params['conversation_id'] = $conversation_id;
    }

    $response = wp_remote_post($base . '/api/v1/chat', array(
        'timeout' => 90,
        'redirection' => 0,
        'sslverify' => true,
        'limit_response_size' => 1048576,
        'headers' => array(
            'Content-Type' => 'application/json; charset=utf-8',
            'Accept' => 'application/json',
            'X-RevOn-Chat-Token' => $token,
            'X-RevOn-Visitor-IP' => $visitor_ip
        ),
        'body' => wp_json_encode($body_params)
    ));

    if (is_wp_error($response)) {
        // Do not return transport details or the backend URL to public visitors.
        return new WP_Error('chat_backend_unavailable', 'The chatbot could not connect right now. Please try again later.', array('status' => 502));
    }

    $http_code = (int) wp_remote_retrieve_response_code($response);
    $payload = json_decode(wp_remote_retrieve_body($response), true);
    if ($http_code < 200 || $http_code >= 300 || !is_array($payload)) {
        if ($http_code === 429) {
            return new WP_Error('chat_rate_limited', 'Too many questions. Please wait a minute and try again.', array('status' => 429));
        }
        if ($http_code === 400 || $http_code === 422) {
            return new WP_Error('chat_invalid_request', 'Please check your question and try again.', array('status' => 400));
        }
        // Hide backend authentication/configuration and internal error details.
        return new WP_Error('chat_backend_error', 'The chatbot is temporarily unavailable. Please try again later.', array('status' => 502));
    }

    if (($payload['status'] ?? '') !== 'success' || !isset($payload['answer']) || !is_string($payload['answer'])) {
        return new WP_Error('chat_invalid_response', 'The chatbot returned an invalid response. Please try again.', array('status' => 502));
    }

    $sources = array();
    if (isset($payload['sources']) && is_array($payload['sources'])) {
        foreach (array_slice($payload['sources'], 0, 5) as $source) {
            if (!is_array($source)) {
                continue;
            }
            $sources[] = array(
                'filename' => isset($source['filename']) && is_scalar($source['filename']) ? sanitize_text_field((string) $source['filename']) : '',
                'page' => isset($source['page']) ? absint($source['page']) : 0,
                'score' => isset($source['score']) && is_numeric($source['score']) ? (float) $source['score'] : null
            );
        }
    }

    return new WP_REST_Response(array(
        'status' => 'success',
        'conversation_id' => isset($payload['conversation_id']) && is_scalar($payload['conversation_id']) ? sanitize_text_field((string) $payload['conversation_id']) : null,
        'answer' => $payload['answer'],
        'sources' => $sources,
        'total_matches' => isset($payload['total_matches']) ? absint($payload['total_matches']) : count($sources),
        'is_policy_query' => !empty($payload['is_policy_query'])
    ), 200);
}

function revon_pcb_public_chat_reset_proxy(WP_REST_Request $request) {
    $conversation_id = $request->get_param('conversation_id');
    if (!is_string($conversation_id) || trim($conversation_id) === '') {
        return new WP_Error('invalid_conversation_id', 'Valid conversation ID is required.', array('status' => 400));
    }
    $conversation_id = sanitize_text_field(trim($conversation_id));

    $base = untrailingslashit((string) get_option('revon_pcb_api_base_url', ''));
    $token = (string) get_option('revon_pcb_chat_token', '');
    if (!$base || !wp_http_validate_url($base) || strpos($base, 'https://') !== 0 || !$token) {
        return new WP_REST_Response(array('status' => 'success', 'cleared' => true, 'local_only' => true), 200);
    }

    $response = wp_remote_post($base . '/api/v1/chat/reset', array(
        'timeout' => 15,
        'sslverify' => true,
        'headers' => array(
            'Content-Type' => 'application/json; charset=utf-8',
            'Accept' => 'application/json',
            'X-RevOn-Chat-Token' => $token
        ),
        'body' => wp_json_encode(array('conversation_id' => $conversation_id))
    ));

    return new WP_REST_Response(array(
        'status' => 'success',
        'conversation_id' => $conversation_id,
        'cleared' => !is_wp_error($response) && wp_remote_retrieve_response_code($response) === 200
    ), 200);
}

// Register admin-ajax streaming handlers for authenticated and unauthenticated visitors
add_action('wp_ajax_nopriv_revon_chat_stream', 'revon_pcb_public_chat_stream');
add_action('wp_ajax_revon_chat_stream', 'revon_pcb_public_chat_stream');

function revon_pcb_public_chat_stream() {
    $raw_input = file_get_contents('php://input');
    $input_data = json_decode($raw_input, true);
    if (!is_array($input_data)) {
        $input_data = $_POST;
    }

    $question = isset($input_data['question']) && is_string($input_data['question']) ? trim($input_data['question']) : '';
    $conversation_id = isset($input_data['conversation_id']) && is_string($input_data['conversation_id']) ? trim($input_data['conversation_id']) : null;

    if ($question === '') {
        wp_send_json_error(array('message' => 'Please enter a question.'), 400);
    }

    $length = function_exists('mb_strlen') ? mb_strlen($question, 'UTF-8') : strlen($question);
    if ($length > 2000) {
        wp_send_json_error(array('message' => 'Your question must be 2,000 characters or fewer.'), 400);
    }

    $rate_key = revon_pcb_public_chat_rate_limit_key();
    $count = (int) get_transient($rate_key);
    if ($count >= 15) {
        wp_send_json_error(array('message' => 'Too many questions. Please wait a minute and try again.'), 429);
    }
    set_transient($rate_key, $count + 1, 70);

    $base = untrailingslashit((string) get_option('revon_pcb_api_base_url', ''));
    $token = (string) get_option('revon_pcb_chat_token', '');
    if (!$base || !wp_http_validate_url($base) || strpos($base, 'https://') !== 0 || !$token) {
        wp_send_json_error(array('message' => 'The chatbot is temporarily unavailable.'), 503);
    }

    if (!function_exists('curl_init')) {
        wp_send_json_error(array('message' => 'cURL extension is required for streaming.'), 500);
    }

    $visitor_ip = isset($_SERVER['REMOTE_ADDR']) ? (string) $_SERVER['REMOTE_ADDR'] : 'unknown';

    // Flush and close all active output buffers before starting SSE stream
    while (ob_get_level()) {
        @ob_end_clean();
    }

    @ini_set('output_buffering', 'off');
    @ini_set('zlib.output_compression', false);
    @ini_set('implicit_flush', 1);

    // Disable proxy and server buffering (Nginx, LiteSpeed, Apache, Cloudflare)
    header('Content-Type: text/event-stream; charset=utf-8');
    header('Cache-Control: no-cache, no-transform');
    header('X-Accel-Buffering: no');
    header('X-LiteSpeed-Buffer: no');
    header('Connection: keep-alive');

    $request_payload = array('question' => $question);
    if ($conversation_id) {
        $request_payload['conversation_id'] = $conversation_id;
    }

    $ch = curl_init($base . '/api/v1/chat/stream');
    curl_setopt_array($ch, array(
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => wp_json_encode($request_payload),
        CURLOPT_HTTPHEADER => array(
            'Content-Type: application/json',
            'Accept: text/event-stream',
            'X-RevOn-Chat-Token: ' . $token,
            'X-RevOn-Visitor-IP: ' . $visitor_ip
        ),
        CURLOPT_WRITEFUNCTION => function ($curl, $chunk) {
            echo $chunk;
            if (ob_get_level()) {
                @ob_flush();
            }
            flush();
            return strlen($chunk);
        },
        CURLOPT_BUFFERSIZE => 128,
        CURLOPT_TIMEOUT => 120,
        CURLOPT_CONNECTTIMEOUT => 15,
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_SSL_VERIFYHOST => 2
    ));

    $exec_result = curl_exec($ch);
    if ($exec_result === false) {
        $error_detail = curl_error($ch);
        echo "event: error\ndata: " . wp_json_encode(array('detail' => 'Stream connection error. Please try again.')) . "\n\n";
        if (ob_get_level()) { @ob_flush(); }
        flush();
    }

    curl_close($ch);
    exit;
}

add_shortcode('revon_policy_chatbot', function () {
    $instance_id = function_exists('wp_unique_id') ? wp_unique_id('revon-policy-chat-') : 'revon-policy-chat-' . wp_rand(1000, 999999);
    $stream_endpoint = admin_url('admin-ajax.php?action=revon_chat_stream');
    $endpoint = rest_url('revon/v1/chat');
    $reset_endpoint = rest_url('revon/v1/chat/reset');
    ob_start();
    ?>
    <div id="<?php echo esc_attr($instance_id); ?>" class="revon-policy-chatbot" aria-label="RevOn Policy Assistant">
        
        <style>
            #<?php echo esc_attr($instance_id); ?> {
                --revon-chat-primary: var(--bs-primary, #ab351a);
                --revon-chat-primary-dark: var(--bs-link-hover-color, #891d03);
                --revon-chat-accent: var(--theme-primary-container, #f56a4a);

                --revon-chat-ink: var(--bs-body-color, #1c1b1b);
                --revon-chat-muted: var(--bs-secondary, #526069);
                --revon-chat-line: var(--theme-border, #E9ECEF);
                --revon-chat-surface: var(--theme-surface, #fcf9f8);
                --revon-chat-soft: var(--theme-surface-muted, #F8F9FA);

                --revon-chat-font-body: var(--bs-font-sans-serif, 'Inter', sans-serif);
                --revon-chat-font-heading: var(--bs-font-heading, 'Outfit', sans-serif);

                font-family: var(--revon-chat-font-body);
                color: var(--revon-chat-ink);
            }

            /* Floating Chat Launcher */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-launcher {
                position: fixed;
                right: 24px;
                bottom: 24px;
                z-index: 99998;
                width: 62px;
                height: 62px;
                border: 0;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(
                    145deg,
                    var(--revon-chat-primary),
                    var(--revon-chat-accent)
                );
                color: #ffffff;
                box-shadow: 0 10px 30px rgba(171, 53, 26, 0.28);
                cursor: pointer;
                transition: transform .18s ease, box-shadow .18s ease;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-launcher:hover {
                transform: translateY(-2px);
                box-shadow: 0 14px 34px rgba(171, 53, 26, 0.36);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-launcher svg {
                width: 28px;
                height: 28px;
            }

            /* Chat Panel */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-panel {
                position: fixed;
                right: 24px;
                bottom: 100px;
                z-index: 99999;
                display: flex;
                flex-direction: column;
                width: min(400px, calc(100vw - 32px));
                height: min(640px, calc(100vh - 128px));
                min-height: 360px;
                overflow: hidden;
                border: 1px solid var(--revon-chat-line);
                border-radius: 20px;
                background: var(--revon-chat-surface);
                box-shadow: 0 22px 70px rgba(28, 27, 27, 0.16);
                opacity: 1;
                transform: translateY(0) scale(1);
                transform-origin: bottom right;
                transition: opacity .18s ease, transform .18s ease;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-panel[hidden] {
                display: none;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-panel.revon-chat-minimized {
                height: auto;
                min-height: 0;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-panel.revon-chat-minimized .revon-chat-content {
                display: none;
            }

            /* Chat Header */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-header {
                flex: 0 0 auto;
                display: flex;
                align-items: center;
                gap: 11px;
                padding: 16px 16px 15px;
                color: #ffffff;
                background: linear-gradient(
                    120deg,
                    var(--revon-chat-primary),
                    var(--revon-chat-accent)
                );
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-avatar {
                width: 40px;
                height: 40px;
                flex: 0 0 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-radius: 13px;
                background: rgba(255, 255, 255, 0.16);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-avatar svg {
                width: 23px;
                height: 23px;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-heading {
                min-width: 0;
                flex: 1;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-title {
                margin: 0;
                color: #ffffff;
                font-family: var(--revon-chat-font-heading);
                font-size: 15px;
                line-height: 1.35;
                font-weight: 700;
                letter-spacing: .1px;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-subtitle {
                margin: 3px 0 0;
                color: rgba(255, 255, 255, 0.90);
                font-size: 11px;
                line-height: 1.35;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-online {
                display: inline-block;
                width: 7px;
                height: 7px;
                margin-right: 4px;
                border-radius: 50%;
                background: #2ECC71;
                vertical-align: 1px;
            }

            /* Controls */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-controls {
                display: flex;
                align-items: center;
                gap: 5px;
            }

            /* Control Buttons (Minimize, Close, Reset) */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-window-control {
                width: 32px;
                height: 32px;
                flex: 0 0 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 0;
                border-radius: 9px;
                background: rgba(255, 255, 255, 0.14);
                color: #ffffff;
                cursor: pointer;
                transition: background .15s ease;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-window-control:hover {
                background: rgba(255, 255, 255, 0.25);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-window-control svg {
                width: 16px;
                height: 16px;
            }

            /* Chat Content */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-content {
                min-height: 0;
                flex: 1;
                display: flex;
                flex-direction: column;
                background: var(--revon-chat-soft);
            }

            /* Message Area */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-log {
                min-height: 0;
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 13px;
                overflow-y: auto;
                padding: 18px 15px 12px;
                scroll-behavior: smooth;
            }

            /* Welcome Message */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-welcome {
                align-self: stretch;
                padding: 14px;
                border: 1px solid var(--revon-chat-line);
                border-radius: 14px 14px 14px 5px;
                background: var(--revon-chat-surface);
                color: var(--revon-chat-ink);
                box-shadow: 0 2px 8px rgba(28, 27, 27, 0.035);
                font-size: 13px;
                line-height: 1.55;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-welcome strong {
                display: block;
                margin-bottom: 4px;
                font-family: var(--revon-chat-font-heading);
                font-size: 14px;
            }

            /* Chat Messages */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-message {
                max-width: 88%;
                padding: 11px 13px;
                border-radius: 14px;
                overflow-wrap: anywhere;
                font-size: 13px;
                line-height: 1.55;
            }

            /* User Message */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-user {
                align-self: flex-end;
                border: 1px solid rgba(171, 53, 26, 0.18);
                border-bottom-right-radius: 5px;
                background: rgba(171, 53, 26, 0.08);
                color: var(--revon-chat-ink);
                white-space: pre-wrap;
            }

            /* Assistant Message */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant {
                align-self: flex-start;
                border: 1px solid var(--revon-chat-line);
                border-bottom-left-radius: 5px;
                background: var(--revon-chat-surface);
                color: var(--revon-chat-ink);
                box-shadow: 0 2px 8px rgba(28, 27, 27, 0.035);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant p {
                margin: 0 0 8px 0;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant p:last-child {
                margin-bottom: 0;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-msg-heading {
                margin: 14px 0 6px 0;
                font-size: 13.5px;
                font-weight: 700;
                color: var(--revon-chat-ink);
                letter-spacing: -0.01em;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant .revon-msg-heading:first-child {
                margin-top: 0;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-msg-list {
                margin: 0 0 9px 0;
                padding-left: 18px;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-msg-list:last-child {
                margin-bottom: 0;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-msg-list li {
                margin: 4px 0;
                line-height: 1.48;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant a {
                color: var(--revon-chat-primary);
                text-decoration: underline;
                text-underline-offset: 2px;
                word-break: break-all;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant a:hover {
                color: #8c2b15;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant strong {
                font-weight: 600;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-chat-assistant code {
                background: rgba(0, 0, 0, 0.05);
                border: 1px solid var(--revon-chat-line);
                padding: 2px 4px;
                border-radius: 4px;
                font-size: 12px;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-source-pill {
                display: inline-block;
                padding: 1px 6px;
                margin: 0 2px;
                font-size: 11px;
                font-weight: 600;
                background: rgba(171, 53, 26, 0.09);
                border: 1px solid rgba(171, 53, 26, 0.28);
                color: var(--revon-chat-primary);
                border-radius: 5px;
            }

            /* Source Citations */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-sources {
                margin-top: 10px;
                padding-top: 9px;
                border-top: 1px solid var(--revon-chat-line);
                font-size: 11px;
                color: var(--revon-chat-muted);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-sources strong {
                display: block;
                margin-bottom: 4px;
                color: var(--revon-chat-ink);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-sources ul {
                margin: 0;
                padding-left: 17px;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-sources li {
                margin: 3px 0;
                overflow-wrap: anywhere;
            }

            /* Typing indicator dots */
            #<?php echo esc_attr($instance_id); ?> .revon-typing-indicator {
                display: inline-flex;
                gap: 4px;
                align-items: center;
                padding: 4px 2px;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-typing-indicator span {
                width: 6px;
                height: 6px;
                background: var(--revon-chat-primary);
                border-radius: 50%;
                animation: revon-pulse 1.3s infinite ease-in-out both;
            }
            #<?php echo esc_attr($instance_id); ?> .revon-typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
            #<?php echo esc_attr($instance_id); ?> .revon-typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
            @keyframes revon-pulse {
                0%, 80%, 100% { transform: scale(0.4); opacity: 0.35; }
                40% { transform: scale(1); opacity: 1; }
            }

            /* Message Composer */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-composer {
                flex: 0 0 auto;
                padding: 11px 13px 12px;
                border-top: 1px solid var(--revon-chat-line);
                background: var(--revon-chat-surface);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-form {
                display: flex;
                align-items: flex-end;
                gap: 8px;
                margin: 0;
            }

            /* Input Field */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-input {
                width: 100%;
                min-height: 45px;
                max-height: 110px;
                resize: vertical;
                box-sizing: border-box;
                padding: 12px 13px;
                border: 1px solid var(--revon-chat-line);
                border-radius: 13px;
                outline: none;
                background: var(--revon-chat-soft);
                color: var(--revon-chat-ink);
                font-family: var(--revon-chat-font-body);
                font-size: 13px;
                line-height: 1.4;
                box-shadow: none;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-input:focus {
                border-color: var(--revon-chat-primary);
                box-shadow: 0 0 0 3px rgba(171, 53, 26, 0.12);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-input::placeholder {
                color: var(--revon-chat-muted);
                opacity: 0.8;
            }

            /* Send Button */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-send {
                width: 45px;
                height: 45px;
                flex: 0 0 45px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
                padding: 0;
                border: 0;
                border-radius: 13px;
                background: var(--revon-chat-primary);
                color: #ffffff;
                cursor: pointer;
                transition: background .18s ease, transform .18s ease;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-send:hover {
                background: var(--revon-chat-primary-dark);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-send:active {
                transform: scale(0.96);
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-send:disabled {
                opacity: 0.55;
                cursor: wait;
            }

            #<?php echo esc_attr($instance_id); ?> .revon-chat-send svg {
                width: 19px;
                height: 19px;
            }

            /* Status and Loading Text */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-status {
                min-height: 16px;
                margin: 7px 2px 0;
                color: var(--revon-chat-muted);
                font-size: 11px;
            }

            /* Footer */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-footer {
                margin-top: 3px;
                color: var(--revon-chat-muted);
                font-size: 10px;
                text-align: center;
            }

            /* Keyboard Accessibility */
            #<?php echo esc_attr($instance_id); ?> .revon-chat-launcher:focus-visible,
            #<?php echo esc_attr($instance_id); ?> .revon-chat-window-control:focus-visible,
            #<?php echo esc_attr($instance_id); ?> .revon-chat-send:focus-visible {
                outline: 3px solid var(--revon-chat-accent);
                outline-offset: 3px;
            }

            /* Mobile Responsive */
            @media (max-width: 520px) {
                #<?php echo esc_attr($instance_id); ?> .revon-chat-launcher {
                    right: 16px;
                    bottom: 16px;
                    width: 58px;
                    height: 58px;
                }

                #<?php echo esc_attr($instance_id); ?> .revon-chat-panel {
                    right: 8px;
                    bottom: 84px;
                    width: calc(100vw - 16px);
                    height: min(680px, calc(100dvh - 100px));
                    min-height: 320px;
                    border-radius: 18px;
                }
            }

            /* Reduced Motion Accessibility */
            @media (prefers-reduced-motion: reduce) {
                #<?php echo esc_attr($instance_id); ?> .revon-chat-launcher,
                #<?php echo esc_attr($instance_id); ?> .revon-chat-panel,
                #<?php echo esc_attr($instance_id); ?> .revon-chat-send {
                    transition: none;
                }
            }
        </style>

        <button type="button" class="revon-chat-launcher" aria-label="Open RevOn Policy Assistant" aria-expanded="false">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8z"/></svg>
        </button>
        <section class="revon-chat-panel" role="dialog" aria-label="RevOn Policy Assistant" aria-modal="false" hidden>
            <header class="revon-chat-header">
                <div class="revon-chat-avatar" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="7" width="16" height="13" rx="3"/><path d="M9 12h.01M15 12h.01M9 16h6M12 3v4M10 3h4"/></svg></div>
                <div class="revon-chat-heading"><h2 class="revon-chat-title">RevOn Policy Assistant</h2><p class="revon-chat-subtitle"><span class="revon-chat-online"></span>Online · Ask about available documents</p></div>
                <div class="revon-chat-controls">
                    <button type="button" class="revon-chat-window-control revon-chat-reset" aria-label="Start new conversation" title="Start new conversation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg></button>
                    <button type="button" class="revon-chat-window-control revon-chat-minimize" aria-label="Minimize chat" title="Minimize"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12h14"/></svg></button>
                    <button type="button" class="revon-chat-window-control revon-chat-close" aria-label="Close chat" title="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="m6 6 12 12M18 6 6 18"/></svg></button>
                </div>
            </header>
            <div class="revon-chat-content">
                <div class="revon-chat-log" role="log" aria-live="polite" aria-relevant="additions">
                    <div class="revon-chat-welcome"><strong>Hello! 👋 Welcome to RevOn.</strong>I'm your Policy Assistant. Ask me a question about the documents currently available, and I'll help you find the relevant information.</div>
                </div>
                <div class="revon-chat-composer">
                    <form class="revon-chat-form">
                        <label class="screen-reader-text" for="<?php echo esc_attr($instance_id); ?>-question">Your question</label>
                        <textarea class="revon-chat-input" id="<?php echo esc_attr($instance_id); ?>-question" name="question" maxlength="2000" rows="1" required placeholder="Type your question..."></textarea>
                        <button type="submit" class="revon-chat-send" aria-label="Send question" title="Send question"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg></button>
                    </form>
                    <div class="revon-chat-status" role="status" aria-live="polite"></div>
                    <div class="revon-chat-footer">Powered by RevOn AI · Answers based on available documents</div>
                </div>
            </div>
        </section>
    </div>
    <script>
    (function () {
        const root = document.getElementById(<?php echo wp_json_encode($instance_id); ?>);
        if (!root) return;
        const endpoint = <?php echo wp_json_encode($endpoint); ?>;
        const streamEndpoint = <?php echo wp_json_encode($stream_endpoint); ?>;
        const resetEndpoint = <?php echo wp_json_encode($reset_endpoint); ?>;

        const launcher = root.querySelector('.revon-chat-launcher');
        const panel = root.querySelector('.revon-chat-panel');
        const minimize = root.querySelector('.revon-chat-minimize');
        const close = root.querySelector('.revon-chat-close');
        const resetBtn = root.querySelector('.revon-chat-reset');
        const form = root.querySelector('.revon-chat-form');
        const input = root.querySelector('.revon-chat-input');
        const button = root.querySelector('.revon-chat-send');
        const log = root.querySelector('.revon-chat-log');
        const status = root.querySelector('.revon-chat-status');

        const STORAGE_KEY = 'revon_chat_cid';
        let activeConversationId = sessionStorage.getItem(STORAGE_KEY) || null;

        function openChat() {
            panel.hidden = false;
            panel.classList.remove('revon-chat-minimized');
            launcher.hidden = true;
            launcher.setAttribute('aria-expanded', 'true');
            window.setTimeout(function () { input.focus(); }, 30);
        }
        function minimizeChat() {
            panel.classList.add('revon-chat-minimized');
            launcher.hidden = false;
            launcher.setAttribute('aria-expanded', 'false');
        }
        function closeChat() {
            panel.hidden = true;
            panel.classList.remove('revon-chat-minimized');
            launcher.hidden = false;
            launcher.setAttribute('aria-expanded', 'false');
            launcher.focus();
        }
        launcher.addEventListener('click', openChat);
        minimize.addEventListener('click', minimizeChat);
        close.addEventListener('click', closeChat);

        // Safe client-side markdown formatter
        function formatMarkdown(raw) {
            if (!raw) return '';

            // Normalize newlines
            let text = raw.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

            // Extract code blocks first to protect their contents
            const codeBlocks = [];
            text = text.replace(/```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g, function (match, lang, code) {
                const placeholder = '___CODEBLOCK_' + codeBlocks.length + '___';
                const langClass = lang ? ' class="language-' + escapeHtml(lang) + '"' : '';
                codeBlocks.push('<pre><code' + langClass + '>' + escapeHtml(code.trim()) + '</code></pre>');
                return '\n\n' + placeholder + '\n\n';
            });

            // Escape HTML special characters
            text = escapeHtml(text);

            // Inline code: `code`
            text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

            // Bold: **text**
            text = text.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');

            // Italic: *text* (excluding bold)
            text = text.replace(/(^|[^\*])\*([^\*]+)\*([^\*]|$)/g, '$1<em>$2</em>$3');

            // Markdown links: [text](url) - strictly sanitize scheme
            text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (match, label, href) {
                const cleanHref = href.trim();
                if (/^(https?:\/\/|mailto:|tel:)/i.test(cleanHref)) {
                    return '<a href="' + cleanHref + '" target="_blank" rel="noopener noreferrer">' + label + '</a>';
                }
                return label;
            });

            // Source pill formatting: [Source X]
            text = text.replace(/\[Source\s+(\d+)\]/g, '<span class="revon-source-pill">[Source $1]</span>');

            // Clean up loose table pipes: lines like "| col 1 | col 2 |"
            text = text.replace(/^\|\s*([^|\n]+(?:\s*\|\s*[^|\n]+)*)\s*\|?$/gm, function (match, inner) {
                if (/^[\s\-:|]+$/.test(inner)) return '';
                return inner.trim();
            });

            // Split into blocks by double newlines
            const rawBlocks = text.split(/\n\s*\n+/);
            const htmlBlocks = [];
            let activeList = null;
            let listItems = [];

            function flushList() {
                if (activeList && listItems.length) {
                    htmlBlocks.push('<' + activeList + ' class="revon-msg-list">' + listItems.join('') + '</' + activeList + '>');
                    activeList = null;
                    listItems = [];
                }
            }

            for (let i = 0; i < rawBlocks.length; i++) {
                let block = rawBlocks[i].trim();
                if (!block) continue;

                // Codeblock placeholder
                if (/^___CODEBLOCK_\d+___$/.test(block)) {
                    flushList();
                    const idx = parseInt(block.replace('___CODEBLOCK_', '').replace('___', ''), 10);
                    htmlBlocks.push(codeBlocks[idx] || '');
                    continue;
                }

                // Headings: ###, ##, #
                const headingMatch = block.match(/^(#{1,4})\s+(.+)$/);
                if (headingMatch) {
                    flushList();
                    const level = headingMatch[1].length;
                    const headingTag = level >= 3 ? 'h4' : 'h3';
                    htmlBlocks.push('<' + headingTag + ' class="revon-msg-heading">' + headingMatch[2].trim() + '</' + headingTag + '>');
                    continue;
                }

                const lines = block.split('\n');

                // Check if block is entirely unordered list (- or *)
                const isBulletList = lines.every(function (line) { return /^\s*[-*]\s+/.test(line); });
                if (isBulletList) {
                    if (activeList && activeList !== 'ul') flushList();
                    activeList = 'ul';
                    for (let j = 0; j < lines.length; j++) {
                        const itemContent = lines[j].replace(/^\s*[-*]\s+/, '').trim();
                        listItems.push('<li>' + itemContent + '</li>');
                    }
                    continue;
                }

                // Check if block is entirely ordered list (1. 2.)
                const isNumList = lines.every(function (line) { return /^\s*\d+\.\s+/.test(line); });
                if (isNumList) {
                    if (activeList && activeList !== 'ol') flushList();
                    activeList = 'ol';
                    for (let j = 0; j < lines.length; j++) {
                        const itemContent = lines[j].replace(/^\s*\d+\.\s+/, '').trim();
                        listItems.push('<li>' + itemContent + '</li>');
                    }
                    continue;
                }

                // Mixed block or regular paragraph
                flushList();

                let hasBullets = false;
                for (let j = 0; j < lines.length; j++) {
                    if (/^\s*[-*]\s+/.test(lines[j])) { hasBullets = true; break; }
                }

                if (hasBullets) {
                    let subHtml = '';
                    let inList = false;
                    for (let j = 0; j < lines.length; j++) {
                        const line = lines[j];
                        if (/^\s*[-*]\s+/.test(line)) {
                            if (!inList) { subHtml += '<ul class="revon-msg-list">'; inList = true; }
                            subHtml += '<li>' + line.replace(/^\s*[-*]\s+/, '').trim() + '</li>';
                        } else {
                            if (inList) { subHtml += '</ul>'; inList = false; }
                            if (line.trim()) {
                                subHtml += '<p>' + line.trim() + '</p>';
                            }
                        }
                    }
                    if (inList) subHtml += '</ul>';
                    htmlBlocks.push(subHtml);
                    continue;
                }

                // Clean up any stray pipes | at beginning or end of line (table artifacts)
                let cleanedBlock = block.replace(/^\|\s*/gm, '').replace(/\s*\|$/gm, '');

                // Regular paragraph
                htmlBlocks.push('<p>' + cleanedBlock.replace(/\n/g, '<br>') + '</p>');
            }

            flushList();
            return htmlBlocks.join('');
        }

        function escapeHtml(str) {
            return str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        }

        function renderSourcesList(sources, parentEl) {
            if (!Array.isArray(sources) || !sources.length) return;
            let wrap = parentEl.querySelector('.revon-chat-sources');
            if (!wrap) {
                wrap = document.createElement('div');
                wrap.className = 'revon-chat-sources';
                const heading = document.createElement('strong');
                heading.textContent = 'Sources';
                wrap.appendChild(heading);
                const list = document.createElement('ul');
                wrap.appendChild(list);
                parentEl.appendChild(wrap);
            }
            const list = wrap.querySelector('ul');
            list.innerHTML = '';
            const seen = new Set();
            sources.forEach(function (source) {
                const filename = source && source.filename ? String(source.filename) : 'Document';
                const page = source && source.page ? Number(source.page) : 0;
                const key = filename + '|' + page;
                if (seen.has(key)) return;
                seen.add(key);
                const li = document.createElement('li');
                li.textContent = filename + (page ? ' — page ' + page : '');
                list.appendChild(li);
            });
        }

        function addMessage(text, role, sources) {
            const item = document.createElement('div');
            item.className = 'revon-chat-message ' + (role === 'user' ? 'revon-chat-user' : 'revon-chat-assistant');
            if (role === 'user') {
                item.textContent = text;
            } else {
                item.innerHTML = '<div class="revon-msg-body">' + formatMarkdown(text) + '</div>';
            }
            log.appendChild(item);
            if (role === 'assistant' && Array.isArray(sources) && sources.length) {
                renderSourcesList(sources, item);
            }
            log.scrollTop = log.scrollHeight;
            return item;
        }

        function createAssistantStreamBubble() {
            const item = document.createElement('div');
            item.className = 'revon-chat-message revon-chat-assistant';
            item.innerHTML = '<div class="revon-msg-body"><div class="revon-typing-indicator"><span></span><span></span><span></span></div></div>';
            log.appendChild(item);
            log.scrollTop = log.scrollHeight;
            return item;
        }

        // New Chat / Session Reset
        if (resetBtn) {
            resetBtn.addEventListener('click', async function () {
                if (activeConversationId) {
                    try {
                        await fetch(resetEndpoint, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ conversation_id: activeConversationId })
                        });
                    } catch (e) {}
                }
                activeConversationId = null;
                sessionStorage.removeItem(STORAGE_KEY);
                const welcome = log.querySelector('.revon-chat-welcome');
                log.innerHTML = '';
                if (welcome) log.appendChild(welcome);
                status.textContent = 'Started a new conversation.';
                window.setTimeout(function () { status.textContent = ''; }, 2500);
                input.focus();
            });
        }

        async function fetchNonStreaming(question, assistantBubble) {
            const bodyEl = assistantBubble.querySelector('.revon-msg-body');
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    credentials: 'same-origin',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify({
                        question: question,
                        conversation_id: activeConversationId
                    })
                });
                const data = await response.json();
                if (!response.ok) {
                    const message = data && data.message ? data.message : 'The chatbot is temporarily unavailable. Please try again.';
                    throw new Error(message);
                }
                if (data.conversation_id) {
                    activeConversationId = data.conversation_id;
                    sessionStorage.setItem(STORAGE_KEY, activeConversationId);
                }
                bodyEl.innerHTML = formatMarkdown(data.answer || 'No answer was returned.');
                if (Array.isArray(data.sources) && data.sources.length) {
                    renderSourcesList(data.sources, assistantBubble);
                }
                status.textContent = '';
            } catch (error) {
                bodyEl.innerHTML = '<p style="color:#e53e3e;">' + (error && error.message ? error.message : 'Unable to load response.') + '</p>';
                status.textContent = '';
            }
        }

        form.addEventListener('submit', async function (event) {
            event.preventDefault();
            const question = input.value.trim();
            if (!question) { status.textContent = 'Please enter a question.'; return; }
            if (question.length > 2000) { status.textContent = 'Your question must be 2,000 characters or fewer.'; return; }
            addMessage(question, 'user');
            input.value = '';
            button.disabled = true;
            status.textContent = 'Connecting…';

            const assistantBubble = createAssistantStreamBubble();
            const bodyEl = assistantBubble.querySelector('.revon-msg-body');
            let streamedText = '';
            let receivedTokens = false;

            try {
                const response = await fetch(streamEndpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
                    body: JSON.stringify({
                        question: question,
                        conversation_id: activeConversationId
                    })
                });

                if (!response.ok || !response.body) {
                    status.textContent = 'Loading response…';
                    await fetchNonStreaming(question, assistantBubble);
                    return;
                }

                status.textContent = '';
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                let currentEvent = 'message';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i].trim();
                        if (!line) {
                            currentEvent = 'message';
                            continue;
                        }

                        if (line.startsWith('event:')) {
                            currentEvent = line.substring(6).trim();
                        } else if (line.startsWith('data:')) {
                            const dataRaw = line.substring(5).trim();
                            let dataObj = null;
                            try {
                                dataObj = JSON.parse(dataRaw);
                            } catch (e) {
                                dataObj = { text: dataRaw };
                            }

                            if (currentEvent === 'start') {
                                if (dataObj && dataObj.conversation_id) {
                                    activeConversationId = dataObj.conversation_id;
                                    sessionStorage.setItem(STORAGE_KEY, activeConversationId);
                                }
                            } else if (currentEvent === 'sources') {
                                if (dataObj && Array.isArray(dataObj.sources)) {
                                    renderSourcesList(dataObj.sources, assistantBubble);
                                }
                            } else if (currentEvent === 'token') {
                                const tok = dataObj && (typeof dataObj.content === 'string' ? dataObj.content : (typeof dataObj.token === 'string' ? dataObj.token : ''));
                                if (tok) {
                                    if (!receivedTokens) {
                                        bodyEl.innerHTML = '';
                                        receivedTokens = true;
                                    }
                                    streamedText += tok;
                                    bodyEl.innerHTML = formatMarkdown(streamedText);
                                    log.scrollTop = log.scrollHeight;
                                }
                            } else if (currentEvent === 'complete') {
                                if (dataObj && dataObj.conversation_id) {
                                    activeConversationId = dataObj.conversation_id;
                                    sessionStorage.setItem(STORAGE_KEY, activeConversationId);
                                }
                                if (dataObj && dataObj.answer && (!receivedTokens || !streamedText.trim())) {
                                    bodyEl.innerHTML = formatMarkdown(dataObj.answer);
                                    streamedText = dataObj.answer;
                                    receivedTokens = true;
                                }
                                if (dataObj && Array.isArray(dataObj.sources)) {
                                    renderSourcesList(dataObj.sources, assistantBubble);
                                }
                            } else if (currentEvent === 'error') {
                                const errMsg = (dataObj && dataObj.detail) ? dataObj.detail : 'An error occurred.';
                                bodyEl.innerHTML = '<p style="color:#e53e3e;">' + errMsg + '</p>';
                            }
                        }
                    }
                }

                if (!receivedTokens || !streamedText.trim()) {
                    await fetchNonStreaming(question, assistantBubble);
                }
            } catch (error) {
                if (!receivedTokens || !streamedText.trim()) {
                    await fetchNonStreaming(question, assistantBubble);
                } else {
                    status.textContent = 'Connection interrupted.';
                }
            } finally {
                button.disabled = false;
                status.textContent = '';
                input.focus();
            }
        });

        input.addEventListener('keydown', function (event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                if (typeof form.requestSubmit === 'function') form.requestSubmit();
            }
        });
    })();
    </script>
    <?php
    return ob_get_clean();
});
